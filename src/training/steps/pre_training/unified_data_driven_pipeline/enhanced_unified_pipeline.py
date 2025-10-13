"""
Enhanced Unified Data-Driven Feature Pipeline

This enhanced version integrates all the advanced components:
- Enhanced VectorBT optimizations
- Economic evaluation with backtesting
- Advanced feature selection from 200+ feature bank
- Complete HTF template system

This provides a comprehensive solution that captures all the logic from
DataDrivenPeriodSelector, DataDrivenInteractionGenerator, 
FeatureLookbackOptimizationComponent, and HTFInteractionTemplates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
from pathlib import Path

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
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

# Import enhanced components
from .enhanced_components.vectorbt_enhancements import (
    EnhancedVectorBTOptimizer, VectorBTOptimizationConfig, create_enhanced_vectorbt_optimizer
)
from .enhanced_components.economic_evaluation import (
    EconomicPeriodEvaluator, EconomicEvaluationConfig, create_economic_evaluator
)
from .enhanced_components.advanced_feature_selection import (
    AdvancedFeatureSelector, FeatureSelectionConfig, create_advanced_feature_selector
)
from .enhanced_components.htf_template_system import (
    HTFInteractionGenerator, HTFTemplateConfig, create_htf_interaction_generator
)

# Import existing components
from .config import UnifiedPipelineConfig, create_default_config
from ..time_series_cv import PurgedEmbargoedWalkForwardCV, create_purged_embargoed_cv
from ..statistical_analysis import StatisticalAnalysisFramework
from ..feature_selection.multi_objective_selector import (
    MultiObjectiveFeatureSelector, 
    create_default_objectives,
    OutOfSampleSharpeObjective,
    DrawdownObjective,
    TurnoverObjective,
    StabilityObjective,
    DiversityObjective,
    MutualInformationObjective,
    ProfitCenteredObjective
)

# Import VectorBT utilities
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    tprint_warning("VectorBT utilities not available, using fallback implementations")

# Import caching and serialization
try:
    from src.feature_generation.core.feature_cache import FeatureCacheService
    from src.utils.serialization_utils import UniversalSerializer, JSONSerializer, PickleSerializer
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    tprint_warning("Caching utilities not available")

# Import matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_batch_matrix_processor,
        safe_matrix_multiply,
        optimize_dataframe,
        matrix_correlation_analysis
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    tprint_warning("Matrix operations not available")

logger = logging.getLogger(__name__)


@dataclass
class EnhancedFeaturePipelineResult:
    """Enhanced result of the unified feature pipeline."""
    # Period optimization results
    optimal_periods: List[int]
    period_scores: Dict[int, float]
    economic_evaluation_results: Optional[Dict[str, Any]] = None
    
    # Feature selection results
    selected_features: List[Any]  # FeatureScore objects
    feature_selection_metrics: Dict[str, Any]
    
    # Interaction generation results
    generated_interactions: List[Any]  # GeneratedInteraction objects
    interaction_metrics: Dict[str, Any]
    
    # HTF template results
    htf_interactions: List[Any]  # GeneratedInteraction objects
    htf_metrics: Dict[str, Any]
    
    # Lookback optimization results
    optimized_lookbacks: Dict[str, int]
    lookback_metrics: Dict[str, Any]
    
    # Overall metrics
    total_processing_time: float
    vectorbt_operations: int
    economic_evaluations: int
    feature_selections: int
    interaction_generations: int
    htf_generations: int
    lookback_optimizations: int
    
    # Success indicators
    success: bool
    error_message: Optional[str] = None


class EnhancedUnifiedDataDrivenPipeline:
    """
    Enhanced Unified Data-Driven Feature Pipeline with all advanced components integrated.
    
    This enhanced version captures all the logic from:
    - DataDrivenPeriodSelector: Advanced period selection with economic evaluation
    - DataDrivenInteractionGenerator: Intelligent feature selection and interaction generation
    - FeatureLookbackOptimizationComponent: Advanced lookback optimization
    - HTFInteractionTemplates: Complete HTF-aware interaction template system
    """
    
    def __init__(self, config: Optional[UnifiedPipelineConfig] = None):
        """Initialize the enhanced unified data-driven pipeline."""
        self.config = config or create_default_config()
        
        # Initialize enhanced components
        self._initialize_enhanced_components()
        
        # Initialize existing components
        self._initialize_existing_components()
        
        # Initialize performance tracking
        self._initialize_performance_tracking()
        
        tprint_info("🚀 Enhanced Unified Data-Driven Pipeline initialized")
        tprint_info(f"📊 Configuration: {self.config}")
    
    def _initialize_enhanced_components(self):
        """Initialize all enhanced components."""
        tprint_debug("Initializing enhanced components")
        
        # Enhanced VectorBT optimizer
        vectorbt_config = VectorBTOptimizationConfig(
            enable_vectorbt=True,
            enable_parallel=True,
            memory_efficient=True,
            optimization_level="high"
        )
        self.vectorbt_optimizer = create_enhanced_vectorbt_optimizer(vectorbt_config)
        
        # Economic period evaluator
        economic_config = EconomicEvaluationConfig(
            min_period=1,
            max_period=50,
            backtest_periods=100,
            min_backtest_periods=50,
            enable_vectorbt=True,
            enable_parallel=True,
            memory_efficient=True,
            min_economic_score=0.4,
            economic_weight=0.6,
            statistical_weight=0.4
        )
        self.economic_evaluator = create_economic_evaluator(economic_config)
        
        # Advanced feature selector
        feature_config = FeatureSelectionConfig(
            target_feature_count=40,
            min_features_per_category=2,
            max_features_per_category=4,
            min_variance=1e-8,
            max_correlation_threshold=0.95,
            min_information_content=0.1,
            enable_parallel_processing=True,
            max_workers=4,
            enable_vectorbt=True,
            enable_diversity_selection=True,
            diversity_threshold=0.3,
            enable_stability_analysis=True,
            stability_window=20
        )
        self.feature_selector = create_advanced_feature_selector(feature_config)
        
        # HTF interaction generator
        htf_config = HTFTemplateConfig(
            enable_vectorbt=True,
            enable_parallel=True,
            memory_efficient=True,
            max_workers=4,
            max_interactions=100,
            utility_threshold=0.1,
            correlation_threshold=0.95,
            enable_htf_aware=True,
            enable_core_templates=True
        )
        self.htf_generator = create_htf_interaction_generator(htf_config)
        
        tprint_success("✅ Enhanced components initialized")
    
    def _initialize_existing_components(self):
        """Initialize existing pipeline components."""
        tprint_debug("Initializing existing components")
        
        # Statistical analysis framework
        self.stats_framework = StatisticalAnalysisFramework()
        
        # Time series CV
        self.cv_splitter = create_purged_embargoed_cv(
            n_splits=self.config.feature_selection.cv_config.n_splits,
            test_size=self.config.feature_selection.cv_config.test_size,
            train_size=self.config.feature_selection.cv_config.train_size,
            purge_fraction=self.config.feature_selection.cv_config.purge_fraction,
            embargo_fraction=self.config.feature_selection.cv_config.embargo_fraction
        )
        
        # Multi-objective feature selector
        self.multi_objective_selector = MultiObjectiveFeatureSelector(
            objectives=create_default_objectives(),
            cv_splitter=self.cv_splitter
        )
        
        # Initialize VectorBT components
        self._initialize_vectorbt_components()
        
        # Initialize caching and serialization
        self._initialize_caching_components()
        
        # Initialize matrix operations
        self._initialize_matrix_components()
        
        tprint_success("✅ Existing components initialized")
    
    def _initialize_vectorbt_components(self):
        """Initialize VectorBT components."""
        try:
            if VECTORBT_AVAILABLE:
                self.vectorbt_rolling_optimizer = VectorBTRollingOptimizer()
                self.unified_vectorization_manager = UnifiedVectorizationManager()
                tprint_success("✅ VectorBT components initialized")
            else:
                tprint_warning("⚠️ VectorBT not available, using fallback implementations")
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT initialization failed: {e}")
    
    def _initialize_caching_components(self):
        """Initialize caching and serialization components."""
        try:
            if CACHING_AVAILABLE:
                self.feature_cache = FeatureCacheService()
                self.universal_serializer = UniversalSerializer()
                self.json_serializer = JSONSerializer()
                self.pickle_serializer = PickleSerializer()
                tprint_success("✅ Caching components initialized")
            else:
                tprint_warning("⚠️ Caching not available")
        except Exception as e:
            tprint_warning(f"⚠️ Caching initialization failed: {e}")
    
    def _initialize_matrix_components(self):
        """Initialize matrix operations components."""
        try:
            if MATRIX_OPS_AVAILABLE:
                self.matrix_operations = get_unified_matrix_operations()
                self.vectorized_processing_core = get_vectorized_processing_core()
                self.batch_matrix_processor = get_batch_matrix_processor()
                tprint_success("✅ Matrix operations components initialized")
            else:
                tprint_warning("⚠️ Matrix operations not available")
        except Exception as e:
            tprint_warning(f"⚠️ Matrix operations initialization failed: {e}")
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking."""
        self.performance_stats = {
            'total_pipeline_runs': 0,
            'successful_pipeline_runs': 0,
            'failed_pipeline_runs': 0,
            'total_execution_time': 0.0,
            'vectorbt_operations': 0,
            'economic_evaluations': 0,
            'feature_selections': 0,
            'interaction_generations': 0,
            'htf_generations': 0,
            'lookback_optimizations': 0
        }
    
    def process(self, data: pd.DataFrame, 
                targets: Optional[pd.Series] = None,
                timeframe: str = "15m") -> EnhancedFeaturePipelineResult:
        """
        Process data through the enhanced unified pipeline.
        
        Args:
            data: Input data with OHLCV columns
            targets: Optional target series for supervised learning
            timeframe: Target timeframe (e.g., "15m", "5m", "1h")
            
        Returns:
            EnhancedFeaturePipelineResult with comprehensive results
        """
        tprint_info("🚀 Starting enhanced unified pipeline processing")
        tprint_info(f"📊 Data shape: {data.shape}, timeframe: {timeframe}")
        
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self._validate_inputs(data, targets):
                return self._create_empty_result(start_time, "Invalid inputs")
            
            # Step 1: Enhanced period optimization with economic evaluation
            tprint_info("Step 1: Enhanced period optimization with economic evaluation")
            period_results = self._enhanced_period_optimization(data, timeframe)
            
            # Step 2: Advanced feature selection from 200+ feature bank
            tprint_info("Step 2: Advanced feature selection from 200+ feature bank")
            feature_selection_results = self._advanced_feature_selection(data, targets)
            
            # Step 3: Generate selected features
            tprint_info("Step 3: Generate selected features")
            selected_features_df = self._generate_selected_features(data, feature_selection_results)
            
            # Step 4: Enhanced interaction generation with VectorBT optimization
            tprint_info("Step 4: Enhanced interaction generation with VectorBT optimization")
            interaction_results = self._enhanced_interaction_generation(selected_features_df, targets)
            
            # Step 5: HTF-aware interaction generation
            tprint_info("Step 5: HTF-aware interaction generation")
            htf_results = self._htf_interaction_generation(data, selected_features_df, targets)
            
            # Step 6: Advanced lookback optimization
            tprint_info("Step 6: Advanced lookback optimization")
            lookback_results = self._advanced_lookback_optimization(data, targets, selected_features_df)
            
            # Step 7: Combine all results
            tprint_info("Step 7: Combine all results")
            combined_results = self._combine_results(
                period_results, feature_selection_results, interaction_results, 
                htf_results, lookback_results
            )
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats.update({
                'total_pipeline_runs': 1,
                'successful_pipeline_runs': 1,
                'total_execution_time': execution_time,
                'vectorbt_operations': self.vectorbt_optimizer.performance_stats['vectorbt_operations'],
                'economic_evaluations': self.economic_evaluator.performance_stats['successful_evaluations'],
                'feature_selections': self.feature_selector.performance_stats['successful_selections'],
                'interaction_generations': len(interaction_results),
                'htf_generations': len(htf_results),
                'lookback_optimizations': len(lookback_results)
            })
            
            tprint_success(f"✅ Enhanced pipeline processing completed in {execution_time:.3f}s")
            tprint_info(f"🏆 Results: {len(combined_results['selected_features'])} features, "
                       f"{len(combined_results['generated_interactions'])} interactions, "
                       f"{len(combined_results['htf_interactions'])} HTF interactions")
            
            return EnhancedFeaturePipelineResult(
                optimal_periods=combined_results['optimal_periods'],
                period_scores=combined_results['period_scores'],
                economic_evaluation_results=combined_results['economic_evaluation_results'],
                selected_features=combined_results['selected_features'],
                feature_selection_metrics=combined_results['feature_selection_metrics'],
                generated_interactions=combined_results['generated_interactions'],
                interaction_metrics=combined_results['interaction_metrics'],
                htf_interactions=combined_results['htf_interactions'],
                htf_metrics=combined_results['htf_metrics'],
                optimized_lookbacks=combined_results['optimized_lookbacks'],
                lookback_metrics=combined_results['lookback_metrics'],
                total_processing_time=execution_time,
                vectorbt_operations=self.performance_stats['vectorbt_operations'],
                economic_evaluations=self.performance_stats['economic_evaluations'],
                feature_selections=self.performance_stats['feature_selections'],
                interaction_generations=self.performance_stats['interaction_generations'],
                htf_generations=self.performance_stats['htf_generations'],
                lookback_optimizations=self.performance_stats['lookback_optimizations'],
                success=True
            )
            
        except Exception as e:
            tprint_error(f"❌ Enhanced pipeline processing failed: {e}")
            return self._create_empty_result(start_time, str(e))
    
    def _validate_inputs(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> bool:
        """Validate input data and parameters."""
        try:
            if data is None or data.empty:
                tprint_error("Data is None or empty")
                return False
            
            if 'close' not in data.columns:
                tprint_error("Data must contain 'close' column")
                return False
            
            if targets is not None and len(targets) != len(data):
                tprint_error("Targets length does not match data length")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"Input validation failed: {e}")
            return False
    
    def _enhanced_period_optimization(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Enhanced period optimization with economic evaluation."""
        tprint_debug("Starting enhanced period optimization")
        
        try:
            # Step 1: Statistical period analysis using VectorBT optimization
            tprint_debug("Step 1: Statistical period analysis")
            periods = list(range(1, 51))  # 1-50 periods for 15m timeframe
            period_analysis = self.vectorbt_optimizer.optimize_period_analysis(data, periods)
            
            # Step 2: Economic significance evaluation
            tprint_debug("Step 2: Economic significance evaluation")
            candidate_periods = [p for p in periods if p in period_analysis and 'error' not in period_analysis[p]]
            economic_evaluation = self.economic_evaluator.evaluate_periods(data, candidate_periods, timeframe)
            
            # Step 3: Combine statistical and economic results
            tprint_debug("Step 3: Combining statistical and economic results")
            combined_scores = self._combine_period_scores(period_analysis, economic_evaluation)
            
            # Step 4: Select optimal periods
            optimal_periods = self._select_optimal_periods(combined_scores)
            
            tprint_success(f"✅ Period optimization completed: {len(optimal_periods)} optimal periods")
            
            return {
                'optimal_periods': optimal_periods,
                'period_scores': combined_scores,
                'economic_evaluation_results': economic_evaluation,
                'statistical_analysis': period_analysis
            }
            
        except Exception as e:
            tprint_error(f"Enhanced period optimization failed: {e}")
            return {
                'optimal_periods': [],
                'period_scores': {},
                'economic_evaluation_results': None,
                'statistical_analysis': {}
            }
    
    def _combine_period_scores(self, statistical_analysis: Dict[int, Dict[str, Any]], 
                              economic_evaluation: Any) -> Dict[int, float]:
        """Combine statistical and economic period scores."""
        try:
            combined_scores = {}
            
            # Extract statistical scores
            for period, analysis in statistical_analysis.items():
                if 'error' not in analysis:
                    # Use sharpe ratio as statistical score
                    sharpe_ratio = analysis.get('sharpe_ratio', 0.0)
                    combined_scores[period] = sharpe_ratio * 0.4  # Statistical weight
            
            # Add economic scores
            if economic_evaluation and hasattr(economic_evaluation, 'period_scores'):
                for period, score in economic_evaluation.period_scores.items():
                    if period in combined_scores:
                        combined_scores[period] += score * 0.6  # Economic weight
                    else:
                        combined_scores[period] = score * 0.6
            
            return combined_scores
            
        except Exception as e:
            tprint_error(f"Period score combination failed: {e}")
            return {}
    
    def _select_optimal_periods(self, period_scores: Dict[int, float], max_periods: int = 8) -> List[int]:
        """Select optimal periods based on combined scores."""
        if not period_scores:
            return []
        
        # Sort by score (descending)
        sorted_periods = sorted(period_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top periods
        optimal_periods = [period for period, score in sorted_periods[:max_periods] if score > 0.1]
        
        return optimal_periods
    
    def _advanced_feature_selection(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> Any:
        """Advanced feature selection from 200+ feature bank."""
        tprint_debug("Starting advanced feature selection")
        
        try:
            # Use the advanced feature selector
            selection_result = self.feature_selector.select_features(data, targets)
            
            if selection_result.success:
                tprint_success(f"✅ Feature selection completed: {len(selection_result.selected_features)} features selected")
                return selection_result
            else:
                tprint_error(f"Feature selection failed: {selection_result.error_message}")
                return None
                
        except Exception as e:
            tprint_error(f"Advanced feature selection failed: {e}")
            return None
    
    def _generate_selected_features(self, data: pd.DataFrame, selection_result: Any) -> pd.DataFrame:
        """Generate features for the selected feature set."""
        tprint_debug("Generating selected features")
        
        try:
            if selection_result is None or not selection_result.success:
                return pd.DataFrame(index=data.index)
            
            # Create feature DataFrame
            features_df = pd.DataFrame(index=data.index)
            
            # Generate features based on selection result
            for feature_score in selection_result.selected_features:
                feature_name = feature_score.feature_name
                
                # Generate the feature based on its name and category
                feature_series = self._generate_single_feature(data, feature_name, feature_score.category)
                
                if feature_series is not None:
                    features_df[feature_name] = feature_series
            
            tprint_success(f"✅ Generated {len(features_df.columns)} features")
            return features_df
            
        except Exception as e:
            tprint_error(f"Feature generation failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def _generate_single_feature(self, data: pd.DataFrame, feature_name: str, category: str) -> Optional[pd.Series]:
        """Generate a single feature based on its name and category."""
        try:
            if 'close' not in data.columns:
                return None
            
            close_prices = data['close']
            feature_series = None
            
            # Generate feature based on category and name
            if category == 'momentum':
                if 'rsi' in feature_name.lower():
                    feature_series = self._calculate_rsi(close_prices, 14)
                elif 'macd' in feature_name.lower():
                    feature_series = self._calculate_macd(close_prices)
                else:
                    feature_series = close_prices.pct_change(20)  # Default momentum
                    
            elif category == 'volatility':
                if 'vol' in feature_name.lower():
                    feature_series = close_prices.rolling(20).std()
                else:
                    feature_series = close_prices.rolling(20).var()
                    
            elif category == 'trend':
                if 'sma' in feature_name.lower():
                    period = int(feature_name.split('_')[-1]) if feature_name.split('_')[-1].isdigit() else 20
                    feature_series = close_prices.rolling(period).mean()
                elif 'ema' in feature_name.lower():
                    period = int(feature_name.split('_')[-1]) if feature_name.split('_')[-1].isdigit() else 12
                    feature_series = close_prices.ewm(span=period).mean()
                else:
                    feature_series = close_prices.rolling(20).mean()  # Default trend
                    
            elif category == 'volume':
                if 'volume' in data.columns:
                    feature_series = data['volume'].rolling(20).mean()
                else:
                    feature_series = pd.Series(0, index=close_prices.index)
                    
            elif category == 'returns':
                if 'return' in feature_name.lower():
                    feature_series = close_prices.pct_change()
                else:
                    feature_series = close_prices.pct_change()
                    
            else:
                # Default feature generation
                feature_series = close_prices.rolling(20).mean()
            
            return feature_series
            
        except Exception as e:
            tprint_debug(f"Feature generation failed for {feature_name}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def _enhanced_interaction_generation(self, features_df: pd.DataFrame, targets: Optional[pd.Series]) -> List[Any]:
        """Enhanced interaction generation with VectorBT optimization."""
        tprint_debug("Starting enhanced interaction generation")
        
        try:
            # Use VectorBT optimizer for interaction generation
            interactions = self.vectorbt_optimizer.optimize_interaction_generation(features_df, targets)
            
            tprint_success(f"✅ Generated {len(interactions)} interactions")
            return interactions
            
        except Exception as e:
            tprint_error(f"Enhanced interaction generation failed: {e}")
            return []
    
    def _htf_interaction_generation(self, data: pd.DataFrame, features_df: pd.DataFrame, 
                                  targets: Optional[pd.Series]) -> List[Any]:
        """HTF-aware interaction generation."""
        tprint_debug("Starting HTF interaction generation")
        
        try:
            # Create simulated HTF features
            htf_features = self._create_htf_features(data)
            
            # Use HTF generator for interaction generation
            htf_interactions = self.htf_generator.generate_interactions(
                htf_features, features_df, targets
            )
            
            tprint_success(f"✅ Generated {len(htf_interactions)} HTF interactions")
            return htf_interactions
            
        except Exception as e:
            tprint_error(f"HTF interaction generation failed: {e}")
            return []
    
    def _create_htf_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create simulated HTF features."""
        try:
            htf_features = {}
            
            if 'close' not in data.columns:
                return htf_features
            
            close_prices = data['close']
            
            # Create simulated HTF features (4h timeframe simulation)
            htf_features['htf_trend'] = close_prices.rolling(16).mean()  # 4h SMA
            htf_features['htf_volatility'] = close_prices.rolling(16).std()  # 4h volatility
            htf_features['htf_momentum'] = close_prices.pct_change(16)  # 4h momentum
            htf_features['htf_regime'] = (close_prices > close_prices.rolling(16).mean()).astype(int)  # 4h regime
            
            return htf_features
            
        except Exception as e:
            tprint_error(f"HTF feature creation failed: {e}")
            return {}
    
    def _advanced_lookback_optimization(self, data: pd.DataFrame, targets: Optional[pd.Series], 
                                      features_df: pd.DataFrame) -> Dict[str, int]:
        """Advanced lookback optimization."""
        tprint_debug("Starting advanced lookback optimization")
        
        try:
            # Use VectorBT optimizer for lookback analysis
            lookback_periods = [5, 10, 15, 20, 25, 30, 40, 50]
            feature_names = list(features_df.columns)
            
            lookback_analysis = self.vectorbt_optimizer.optimize_lookback_analysis(
                data, feature_names, lookback_periods
            )
            
            # Select optimal lookback for each feature
            optimized_lookbacks = {}
            for feature_name, analysis in lookback_analysis.items():
                if analysis:
                    # Find best lookback period
                    best_lookback = max(analysis.keys(), key=lambda k: analysis[k].get('stability', 0))
                    optimized_lookbacks[feature_name] = best_lookback
            
            tprint_success(f"✅ Optimized lookbacks for {len(optimized_lookbacks)} features")
            return optimized_lookbacks
            
        except Exception as e:
            tprint_error(f"Advanced lookback optimization failed: {e}")
            return {}
    
    def _combine_results(self, period_results: Dict[str, Any], feature_selection_results: Any,
                        interaction_results: List[Any], htf_results: List[Any], 
                        lookback_results: Dict[str, int]) -> Dict[str, Any]:
        """Combine all pipeline results."""
        try:
            return {
                'optimal_periods': period_results.get('optimal_periods', []),
                'period_scores': period_results.get('period_scores', {}),
                'economic_evaluation_results': period_results.get('economic_evaluation_results'),
                'selected_features': feature_selection_results.selected_features if feature_selection_results else [],
                'feature_selection_metrics': feature_selection_results.quality_metrics if feature_selection_results else {},
                'generated_interactions': interaction_results,
                'interaction_metrics': self._calculate_interaction_metrics(interaction_results),
                'htf_interactions': htf_results,
                'htf_metrics': self._calculate_interaction_metrics(htf_results),
                'optimized_lookbacks': lookback_results,
                'lookback_metrics': self._calculate_lookback_metrics(lookback_results)
            }
            
        except Exception as e:
            tprint_error(f"Result combination failed: {e}")
            return {
                'optimal_periods': [],
                'period_scores': {},
                'economic_evaluation_results': None,
                'selected_features': [],
                'feature_selection_metrics': {},
                'generated_interactions': [],
                'interaction_metrics': {},
                'htf_interactions': [],
                'htf_metrics': {},
                'optimized_lookbacks': {},
                'lookback_metrics': {}
            }
    
    def _calculate_interaction_metrics(self, interactions: List[Any]) -> Dict[str, Any]:
        """Calculate metrics for generated interactions."""
        if not interactions:
            return {}
        
        try:
            return {
                'total_interactions': len(interactions),
                'average_utility_score': np.mean([i.utility_score for i in interactions]),
                'max_utility_score': max(i.utility_score for i in interactions),
                'min_utility_score': min(i.utility_score for i in interactions),
                'interaction_types': list(set(i.interaction_type for i in interactions)),
                'unique_parent_features': len(set(f for i in interactions for f in i.parent_features))
            }
        except:
            return {}
    
    def _calculate_lookback_metrics(self, lookback_results: Dict[str, int]) -> Dict[str, Any]:
        """Calculate metrics for lookback optimization."""
        if not lookback_results:
            return {}
        
        try:
            return {
                'total_features_optimized': len(lookback_results),
                'average_lookback': np.mean(list(lookback_results.values())),
                'min_lookback': min(lookback_results.values()),
                'max_lookback': max(lookback_results.values()),
                'lookback_distribution': dict(pd.Series(list(lookback_results.values())).value_counts())
            }
        except:
            return {}
    
    def _create_empty_result(self, start_time: float, error_message: str) -> EnhancedFeaturePipelineResult:
        """Create empty result for failed processing."""
        return EnhancedFeaturePipelineResult(
            optimal_periods=[],
            period_scores={},
            economic_evaluation_results=None,
            selected_features=[],
            feature_selection_metrics={},
            generated_interactions=[],
            interaction_metrics={},
            htf_interactions=[],
            htf_metrics={},
            optimized_lookbacks={},
            lookback_metrics={},
            total_processing_time=time.time() - start_time,
            vectorbt_operations=0,
            economic_evaluations=0,
            feature_selections=0,
            interaction_generations=0,
            htf_generations=0,
            lookback_optimizations=0,
            success=False,
            error_message=error_message
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add component stats
        stats['vectorbt_optimizer'] = self.vectorbt_optimizer.get_performance_stats()
        stats['economic_evaluator'] = self.economic_evaluator.get_performance_stats()
        stats['feature_selector'] = self.feature_selector.get_performance_stats()
        stats['htf_generator'] = self.htf_generator.get_performance_stats()
        
        return stats
    
    def reset_stats(self):
        """Reset all performance statistics."""
        self.performance_stats = {
            'total_pipeline_runs': 0,
            'successful_pipeline_runs': 0,
            'failed_pipeline_runs': 0,
            'total_execution_time': 0.0,
            'vectorbt_operations': 0,
            'economic_evaluations': 0,
            'feature_selections': 0,
            'interaction_generations': 0,
            'htf_generations': 0,
            'lookback_optimizations': 0
        }
        
        # Reset component stats
        self.vectorbt_optimizer.reset_stats()
        self.economic_evaluator.reset_stats()
        self.feature_selector.reset_stats()
        self.htf_generator.reset_stats()


def create_enhanced_unified_pipeline(config: Optional[UnifiedPipelineConfig] = None) -> EnhancedUnifiedDataDrivenPipeline:
    """Create an enhanced unified data-driven pipeline with default configuration."""
    return EnhancedUnifiedDataDrivenPipeline(config)