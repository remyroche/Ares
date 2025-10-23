"""
Enhanced Data-Driven Interaction Generator with Intelligent Feature Selection

This module provides a comprehensive data-driven approach to generating interaction features
by first intelligently selecting 40-ish features from the full feature bank (200+ features),
then generating interactions between them.

Key Features:
- Data-driven feature pre-selection from full feature bank
- Ensures at least 3 features per category with different aspects
- Comprehensive interaction generation with VectorBT optimization
- Intelligent quality filtering and ranking
- Performance monitoring and statistics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
import time
import warnings

# Import tprint for comprehensive logging
try:
    from tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)

# Import the data-driven feature selector
try:
    from .data_driven_feature_selector import (
        DataDrivenFeatureSelector,
        FeatureSelectionConfig,
        FeatureSelectionResult
    )
    FEATURE_SELECTOR_AVAILABLE = True
except ImportError:
    FEATURE_SELECTOR_AVAILABLE = False
    DataDrivenFeatureSelector = None
    FeatureSelectionConfig = None
    FeatureSelectionResult = None

# Import the base interaction generator
try:
    from .data_driven_interaction_generator import (
        DataDrivenInteractionGenerator,
        InteractionResult,
        InteractionType,
        EnhancedInteractionConfig
    )
    INTERACTION_GENERATOR_AVAILABLE = True
except ImportError:
    INTERACTION_GENERATOR_AVAILABLE = False
    DataDrivenInteractionGenerator = None
    InteractionResult = None
    InteractionType = None
    EnhancedInteractionConfig = None

logger = logging.getLogger(__name__)

@dataclass
class EnhancedDataDrivenConfig:
    """Enhanced configuration for data-driven interaction generation."""
    # Feature selection settings
    target_feature_count: int = 40
    min_features_per_category: int = 2
    max_features_per_category: int = 4

    # Interaction generation settings
    max_interactions: int = 100
    utility_threshold: float = 0.1
    correlation_threshold: float = 0.95

    # Quality settings
    min_variance: float = 1e-8
    max_correlation_threshold: float = 0.95
    min_information_content: float = 0.1

    # Performance settings
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    enable_batch_processing: bool = True
    memory_efficient: bool = True
    max_workers: int = 4

    # Category weights for feature selection
    category_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.category_weights is None:
            self.category_weights = {
                'momentum': 1.0,
                'volatility': 1.0,
                'trend': 1.0,
                'oscillator': 1.0,
                'volume': 1.0,
                'returns': 1.0,
                'cross_timeframe': 1.2,
                'microstructure': 1.1,
                'entropy': 0.9,
                'support_resistance': 0.9,
                'candlestick_pattern': 0.8,
                'time': 0.7,
                'order_flow': 1.0,
                'regime': 1.0,
                'acceleration': 1.0,
                'advanced_statistical': 1.0,
                'spectral_wavelet': 0.9
            }

@dataclass
class EnhancedInteractionResult:
    """Enhanced result containing both feature selection and interaction generation results."""
    # Feature selection results
    selected_features: List[Any]  # FeatureScore objects
    feature_selection_metrics: Dict[str, Any]

    # Interaction generation results
    interactions: List[InteractionResult]
    interaction_metrics: Dict[str, Any]

    # Overall metrics
    total_processing_time: float
    total_features_analyzed: int
    final_feature_count: int
    final_interaction_count: int

    # Metadata
    config: Dict[str, Any]
    metadata: Dict[str, Any] = None

class EnhancedDataDrivenInteractionGenerator:
    """
    Enhanced data-driven interaction generator with intelligent feature pre-selection.

    This class combines intelligent feature selection from the full feature bank
    with comprehensive interaction generation, ensuring diversity and quality.
    """

    def __init__(self, config: Optional[EnhancedDataDrivenConfig] = None):
        """
        Initialize the enhanced data-driven interaction generator.

        Args:
            config: Configuration for the enhanced generator
        """
        self.config = config or EnhancedDataDrivenConfig()

        # Initialize feature selector
        if FEATURE_SELECTOR_AVAILABLE:
            feature_selector_config = FeatureSelectionConfig(
                target_feature_count=self.config.target_feature_count,
                min_features_per_category=self.config.min_features_per_category,
                max_features_per_category=self.config.max_features_per_category,
                min_variance=self.config.min_variance,
                max_correlation_threshold=self.config.max_correlation_threshold,
                min_information_content=self.config.min_information_content,
                enable_parallel_processing=self.config.enable_parallel,
                max_workers=self.config.max_workers,
                enable_vectorbt=self.config.enable_vectorbt,
                category_weights=self.config.category_weights
            )
            self.feature_selector = DataDrivenFeatureSelector(feature_selector_config)
        else:
            self.feature_selector = None
            logger.warning("Feature selector not available, using basic feature selection")

        # Initialize interaction generator
        if INTERACTION_GENERATOR_AVAILABLE:
            interaction_config = EnhancedInteractionConfig(
                max_interactions=self.config.max_interactions,
                utility_threshold=self.config.utility_threshold,
                correlation_threshold=self.config.correlation_threshold,
                enable_vectorbt=self.config.enable_vectorbt,
                enable_parallel=self.config.enable_parallel,
                enable_batch_processing=self.config.enable_batch_processing,
                memory_efficient=self.config.memory_efficient,
                max_workers=self.config.max_workers
            )
            self.interaction_generator = DataDrivenInteractionGenerator(config=interaction_config)
        else:
            self.interaction_generator = None
            logger.warning("Interaction generator not available, using basic implementation")

        # Performance tracking
        self.performance_stats = {
            'total_processing_time': 0.0,
            'feature_selection_time': 0.0,
            'interaction_generation_time': 0.0,
            'total_features_analyzed': 0,
            'selected_features_count': 0,
            'generated_interactions_count': 0,
            'feature_categories_used': 0,
            'interaction_types_used': 0
        }

        logger.info(f"✅ Enhanced data-driven interaction generator initialized")
        logger.info(f"📊 Target features: {self.config.target_feature_count}")
        logger.info(f"📊 Max interactions: {self.config.max_interactions}")
        logger.info(f"📊 Feature selector available: {FEATURE_SELECTOR_AVAILABLE}")
        logger.info(f"📊 Interaction generator available: {INTERACTION_GENERATOR_AVAILABLE}")

    def generate_interactions(self,
                            data: pd.DataFrame,
                            targets: Optional[pd.Series] = None,
                            available_categories: Optional[List[str]] = None) -> EnhancedInteractionResult:
        """
        Generate interactions using data-driven feature selection and interaction generation.

        Args:
            data: Input data for feature generation
            targets: Target variable for relevance scoring
            available_categories: Specific categories to consider (None = all)

        Returns:
            EnhancedInteractionResult with selected features and generated interactions
        """
        tprint("🚀 Starting enhanced data-driven interaction generation")
        tprint(f"📊 Input data shape: {data.shape}")

        try:
            start_time = time.time()

            # Validate inputs
            if data is None or data.empty:
                tprint("❌ ERROR: Input data is None or empty")
                return self._create_empty_result()

            if not isinstance(data, pd.DataFrame):
                tprint("❌ ERROR: Input data must be a pandas DataFrame")
                return self._create_empty_result()

            tprint(f"✅ Input validation passed: {len(data.columns)} features, {len(data)} samples")

            # Step 1: Select features using data-driven approach
            tprint("🎯 Step 1: Data-driven feature selection...")
            feature_selection_start = time.time()
            try:
                selected_features_result = self._select_features(data, targets, available_categories)
                feature_selection_time = time.time() - feature_selection_start

                if selected_features_result.metadata.get('error', False):
                    tprint("❌ ERROR: Feature selection failed")
                    return self._create_empty_result()

                tprint(f"✅ Feature selection completed in {feature_selection_time:.2f}s")
                tprint(f"📊 Selected {len(selected_features_result.selected_features)} features")
                tprint(f"📊 Categories used: {list(selected_features_result.category_distribution.keys())}")
            except Exception as e:
                tprint(f"❌ ERROR: Feature selection failed: {e}")
                return self._create_empty_result()

            # Step 2: Generate features for selected feature set
            tprint("🔧 Step 2: Generating features for selected feature set...")
            try:
                selected_features_df = self._generate_selected_features(data, selected_features_result)
                tprint(f"✅ Generated feature DataFrame: {selected_features_df.shape}")
            except Exception as e:
                tprint(f"❌ ERROR: Feature generation failed: {e}")
                return self._create_empty_result()

            # Step 3: Generate interactions between selected features
            tprint("⚡ Step 3: Generating interactions between selected features...")
            interaction_generation_start = time.time()
            try:
                interactions = self._generate_interactions(selected_features_df, targets)
                interaction_generation_time = time.time() - interaction_generation_start

                tprint(f"✅ Interaction generation completed in {interaction_generation_time:.2f}s")
                tprint(f"📊 Generated {len(interactions)} interactions")
            except Exception as e:
                tprint(f"❌ ERROR: Interaction generation failed: {e}")
                return self._create_empty_result()

            # Step 4: Calculate overall metrics
            total_processing_time = time.time() - start_time

            # Update performance stats
            self.performance_stats.update({
                'total_processing_time': total_processing_time,
                'feature_selection_time': feature_selection_time,
                'interaction_generation_time': interaction_generation_time,
                'total_features_analyzed': selected_features_result.total_features_analyzed,
                'selected_features_count': len(selected_features_result.selected_features),
                'generated_interactions_count': len(interactions),
                'feature_categories_used': len(selected_features_result.category_distribution),
                'interaction_types_used': len(set(i.interaction_type for i in interactions))
            })

            # Create enhanced result
            tprint("📊 Step 4: Creating enhanced result...")
            try:
                result = EnhancedInteractionResult(
                    selected_features=selected_features_result.selected_features,
                    feature_selection_metrics=selected_features_result.quality_metrics,
                    interactions=interactions,
                    interaction_metrics=self._calculate_interaction_metrics(interactions),
                    total_processing_time=total_processing_time,
                    total_features_analyzed=selected_features_result.total_features_analyzed,
                    final_feature_count=len(selected_features_result.selected_features),
                    final_interaction_count=len(interactions),
                    config=self.config.__dict__,
                    metadata={
                        'feature_selection_result': selected_features_result,
                        'performance_stats': self.performance_stats.copy()
                    }
                )

                tprint(f"✅ Enhanced interaction generation completed in {total_processing_time:.2f}s")
                tprint(f"📊 Final results: {result.final_feature_count} features, {result.final_interaction_count} interactions")
                tprint(f"📊 Performance: {self.performance_stats}")

                return result

            except Exception as e:
                tprint(f"❌ ERROR: Result creation failed: {e}")
                return self._create_empty_result()

        except Exception as e:
            tprint(f"❌ CRITICAL ERROR: Enhanced interaction generation failed: {e}")
            logger.exception("Critical error in generate_interactions")
            return self._create_empty_result()

    def _create_empty_result(self) -> EnhancedInteractionResult:
        """Create an empty result for error cases."""
        return EnhancedInteractionResult(
            selected_features=[],
            feature_selection_metrics={},
            interactions=[],
            interaction_metrics={},
            total_processing_time=0.0,
            total_features_analyzed=0,
            final_feature_count=0,
            final_interaction_count=0,
            config=self.config.__dict__,
            metadata={'error': True}
        )

    def _select_features(self,
                        data: pd.DataFrame,
                        targets: Optional[pd.Series],
                        available_categories: Optional[List[str]]) -> FeatureSelectionResult:
        """Select features using data-driven approach."""
        if self.feature_selector:
            return self.feature_selector.select_features(data, targets, available_categories)
        else:
            # Fallback to basic feature selection
            return self._basic_feature_selection(data, targets)

    def _basic_feature_selection(self,
                                data: pd.DataFrame,
                                targets: Optional[pd.Series]) -> FeatureSelectionResult:
        """Basic feature selection as fallback."""
        # This is a simplified fallback - in practice, you'd want a more robust implementation
        from .data_driven_feature_selector import FeatureSelectionResult, FeatureScore

        # Select basic features
        basic_features = []
        feature_names = []

        # Price-based features (excluding raw OHLCV as requested)
        if 'close' in data.columns:
            feature_names.extend([
                'close_return', 'close_log_return', 'close_volatility_20',
                'close_sma_20', 'close_ema_12', 'close_rsi_14'
            ])

        # Volume features
        if 'volume' in data.columns:
            feature_names.extend(['volume_sma_20', 'volume_ratio', 'volume_momentum'])

        # Create basic feature scores
        for i, name in enumerate(feature_names[:self.config.target_feature_count]):
            score = FeatureScore(
                feature_name=name,
                category='basic',
                aspect_type='general',
                score=1.0 - (i * 0.01),  # Decreasing scores
                variance=1.0,
                correlation_with_target=0.0,
                information_content=0.5,
                uniqueness_score=0.5
            )
            basic_features.append(score)

        return FeatureSelectionResult(
            selected_features=basic_features,
            category_distribution={'basic': len(basic_features)},
            aspect_distribution={'general': len(basic_features)},
            total_features_analyzed=len(feature_names),
            selection_time=0.0,
            quality_metrics={'average_score': 0.5}
        )

    def _generate_selected_features(self,
                                   data: pd.DataFrame,
                                   selection_result: FeatureSelectionResult) -> pd.DataFrame:
        """Generate features for the selected feature set."""
        # This would typically use the feature bank to generate the selected features
        # For now, we'll create a basic implementation

        features_df = pd.DataFrame(index=data.index)

        # Generate basic features based on selection result
        for feature_score in selection_result.selected_features:
            feature_name = feature_score.feature_name

            # Generate the feature based on its name and category
            if 'close' in feature_name and 'close' in data.columns:
                if 'return' in feature_name:
                    features_df[feature_name] = data['close'].pct_change()
                elif 'log_return' in feature_name:
                    features_df[feature_name] = np.log(data['close'] / data['close'].shift(1))
                elif 'volatility' in feature_name:
                    period = int(feature_name.split('_')[-1]) if feature_name.split('_')[-1].isdigit() else 20
                    features_df[feature_name] = data['close'].rolling(period).std()
                elif 'sma' in feature_name:
                    period = int(feature_name.split('_')[-1]) if feature_name.split('_')[-1].isdigit() else 20
                    features_df[feature_name] = data['close'].rolling(period).mean()
                elif 'ema' in feature_name:
                    period = int(feature_name.split('_')[-1]) if feature_name.split('_')[-1].isdigit() else 12
                    features_df[feature_name] = data['close'].ewm(span=period).mean()
                elif 'rsi' in feature_name:
                    period = int(feature_name.split('_')[-1]) if feature_name.split('_')[-1].isdigit() else 14
                    features_df[feature_name] = self._calculate_rsi(data['close'], period)

            elif 'volume' in feature_name and 'volume' in data.columns:
                if 'sma' in feature_name:
                    period = int(feature_name.split('_')[-1]) if feature_name.split('_')[-1].isdigit() else 20
                    features_df[feature_name] = data['volume'].rolling(period).mean()
                elif 'ratio' in feature_name:
                    features_df[feature_name] = data['volume'] / data['volume'].rolling(20).mean()
                elif 'momentum' in feature_name:
                    features_df[feature_name] = data['volume'].pct_change()

        return features_df

    def _generate_interactions(self,
                             features: pd.DataFrame,
                             targets: Optional[pd.Series]) -> List[InteractionResult]:
        """Generate interactions between selected features."""
        if self.interaction_generator:
            return self.interaction_generator.generate_interactions(features, targets)
        else:
            # Fallback to basic interaction generation
            return self._basic_interaction_generation(features, targets)

    def _basic_interaction_generation(self,
                                    features: pd.DataFrame,
                                    targets: Optional[pd.Series]) -> List[InteractionResult]:
        """Basic interaction generation as fallback."""
        # This is a simplified fallback - in practice, you'd want a more robust implementation
        interactions = []

        # Generate basic interactions including log interactions
        feature_names = list(features.columns)

        # Basic arithmetic interactions
        for i, feat1 in enumerate(feature_names):
            for feat2 in feature_names[i+1:]:
                # Product interaction
                try:
                    product = features[feat1] * features[feat2]
                    if not product.isna().all():
                        utility_score = abs(product.corr(targets)) if targets is not None else product.var()
                        if utility_score > self.config.utility_threshold:
                            interaction = InteractionResult(
                                feature_name=f"product_{feat1}_{feat2}",
                                feature_series=product,
                                parent_features=[feat1, feat2],
                                interaction_type="product",
                                utility_score=utility_score,
                                metadata={}
                            )
                            interactions.append(interaction)
                except:
                    continue

                # Log product interaction
                try:
                    # Ensure positive values for log transformation
                    feat1_safe = features[feat1].copy()
                    feat2_safe = features[feat2].copy()
                    feat1_safe = np.where(feat1_safe <= 0, np.abs(feat1_safe) + 1e-8, feat1_safe)
                    feat2_safe = np.where(feat2_safe <= 0, np.abs(feat2_safe) + 1e-8, feat2_safe)

                    log_product = np.log(feat1_safe) * np.log(feat2_safe)
                    log_product = pd.Series(log_product, index=features.index)

                    if not log_product.isna().all():
                        utility_score = abs(log_product.corr(targets)) if targets is not None else log_product.var()
                        if utility_score > self.config.utility_threshold:
                            interaction = InteractionResult(
                                feature_name=f"log_product_{feat1}_{feat2}",
                                feature_series=log_product,
                                parent_features=[feat1, feat2],
                                interaction_type="log_product",
                                utility_score=utility_score,
                                metadata={}
                            )
                            interactions.append(interaction)
                except:
                    continue

                # Log return interaction (if features look like prices)
                try:
                    if 'close' in feat1.lower() or 'price' in feat1.lower():
                        log_ret1 = np.log(features[feat1] / features[feat1].shift(1))
                        log_ret2 = np.log(features[feat2] / features[feat2].shift(1))
                        log_ret1 = log_ret1.fillna(0)
                        log_ret2 = log_ret2.fillna(0)

                        log_return_product = log_ret1 * log_ret2
                        log_return_product = pd.Series(log_return_product, index=features.index)

                        if not log_return_product.isna().all():
                            utility_score = abs(log_return_product.corr(targets)) if targets is not None else log_return_product.var()
                            if utility_score > self.config.utility_threshold:
                                interaction = InteractionResult(
                                    feature_name=f"log_return_product_{feat1}_{feat2}",
                                    feature_series=log_return_product,
                                    parent_features=[feat1, feat2],
                                    interaction_type="log_return_product",
                                    utility_score=utility_score,
                                    metadata={}
                                )
                                interactions.append(interaction)
                except:
                    continue

        # Sort by utility score
        interactions.sort(key=lambda x: x.utility_score, reverse=True)

        return interactions[:self.config.max_interactions]

    def _calculate_interaction_metrics(self, interactions: List[InteractionResult]) -> Dict[str, Any]:
        """Calculate metrics for generated interactions."""
        if not interactions:
            return {}

        return {
            'total_interactions': len(interactions),
            'average_utility_score': np.mean([i.utility_score for i in interactions]),
            'max_utility_score': max(i.utility_score for i in interactions),
            'min_utility_score': min(i.utility_score for i in interactions),
            'interaction_types': list(set(i.interaction_type for i in interactions)),
            'unique_parent_features': len(set(f for i in interactions for f in i.parent_features))
        }

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
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

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return self.performance_stats.copy()

    def reset_stats(self):
        """Reset all performance statistics."""
        self.performance_stats = {
            'total_processing_time': 0.0,
            'feature_selection_time': 0.0,
            'interaction_generation_time': 0.0,
            'total_features_analyzed': 0,
            'selected_features_count': 0,
            'generated_interactions_count': 0,
            'feature_categories_used': 0,
            'interaction_types_used': 0
        }

def create_enhanced_data_driven_interaction_generator(
    config: Optional[EnhancedDataDrivenConfig] = None
) -> EnhancedDataDrivenInteractionGenerator:
    """Create an enhanced data-driven interaction generator with default configuration."""
    return EnhancedDataDrivenInteractionGenerator(config)
