"""
Data-Driven Interaction Feature Generator

This module provides a comprehensive data-driven approach to generating
interaction features by exploring different interaction types and combinations
based on data characteristics and feature bank analysis.

Key Features:
- Data-driven interaction type selection
- Automatic parameter optimization
- VectorBT-optimized computations
- Comprehensive interaction exploration
- Memory-efficient processing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import logging
import warnings
from itertools import combinations, product
import time

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
    )
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    rolling_quantile = None
    rolling_skew = None
    rolling_kurt = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

logger = logging.getLogger(__name__)


@dataclass
class InteractionType:
    """Represents a type of interaction feature."""
    name: str
    function: Callable
    description: str
    complexity: int  # 1-5 scale
    vectorbt_optimized: bool = True
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class InteractionResult:
    """Result of interaction feature generation."""
    feature_name: str
    feature_series: pd.Series
    parent_features: List[str]
    interaction_type: str
    utility_score: float
    metadata: Dict[str, Any]


class DataDrivenInteractionGenerator:
    """
    Generates interaction features using a data-driven approach.
    
    This class explores different interaction types and combinations
    based on data characteristics and feature bank analysis.
    """
    
    def __init__(self, 
                 max_interactions: int = 100,
                 utility_threshold: float = 0.1,
                 correlation_threshold: float = 0.95,
                 enable_vectorbt: bool = True):
        """
        Initialize the data-driven interaction generator.
        
        Args:
            max_interactions: Maximum number of interactions to generate
            utility_threshold: Minimum utility score for feature selection
            correlation_threshold: Maximum correlation for feature filtering
            enable_vectorbt: Whether to use VectorBT optimization
        """
        self.max_interactions = max_interactions
        self.utility_threshold = utility_threshold
        self.correlation_threshold = correlation_threshold
        self.enable_vectorbt = enable_vectorbt and VECTORBT_AVAILABLE
        
        # Initialize interaction types
        self.interaction_types = self._initialize_interaction_types()
        
        logger.info(f"✅ Data-driven interaction generator initialized")
        logger.info(f"📊 Max interactions: {max_interactions}")
        logger.info(f"📊 Utility threshold: {utility_threshold}")
        logger.info(f"📊 VectorBT enabled: {self.enable_vectorbt}")
    
    def _initialize_interaction_types(self) -> Dict[str, InteractionType]:
        """Initialize available interaction types."""
        interaction_types = {}
        
        # Basic arithmetic interactions
        interaction_types['product'] = InteractionType(
            name='product',
            function=self._product_interaction,
            description='Multiplication of two features',
            complexity=1,
            vectorbt_optimized=True
        )
        
        interaction_types['ratio'] = InteractionType(
            name='ratio',
            function=self._ratio_interaction,
            description='Division of two features',
            complexity=1,
            vectorbt_optimized=True
        )
        
        interaction_types['difference'] = InteractionType(
            name='difference',
            function=self._difference_interaction,
            description='Subtraction of two features',
            complexity=1,
            vectorbt_optimized=True
        )
        
        interaction_types['sum'] = InteractionType(
            name='sum',
            function=self._sum_interaction,
            description='Addition of two features',
            complexity=1,
            vectorbt_optimized=True
        )
        
        # Advanced interactions
        interaction_types['correlation'] = InteractionType(
            name='correlation',
            function=self._correlation_interaction,
            description='Rolling correlation between features',
            complexity=3,
            vectorbt_optimized=True,
            parameters={'window': 20}
        )
        
        interaction_types['covariance'] = InteractionType(
            name='covariance',
            function=self._covariance_interaction,
            description='Rolling covariance between features',
            complexity=3,
            vectorbt_optimized=True,
            parameters={'window': 20}
        )
        
        interaction_types['zscore_product'] = InteractionType(
            name='zscore_product',
            function=self._zscore_product_interaction,
            description='Product of z-scored features',
            complexity=2,
            vectorbt_optimized=True
        )
        
        interaction_types['rank_correlation'] = InteractionType(
            name='rank_correlation',
            function=self._rank_correlation_interaction,
            description='Rank correlation between features',
            complexity=3,
            vectorbt_optimized=True,
            parameters={'window': 20}
        )
        
        # Polynomial interactions
        interaction_types['quadratic'] = InteractionType(
            name='quadratic',
            function=self._quadratic_interaction,
            description='Quadratic transformation of feature',
            complexity=2,
            vectorbt_optimized=True
        )
        
        interaction_types['cubic'] = InteractionType(
            name='cubic',
            function=self._cubic_interaction,
            description='Cubic transformation of feature',
            complexity=2,
            vectorbt_optimized=True
        )
        
        # Statistical interactions
        interaction_types['skewness'] = InteractionType(
            name='skewness',
            function=self._skewness_interaction,
            description='Rolling skewness of feature',
            complexity=3,
            vectorbt_optimized=True,
            parameters={'window': 20}
        )
        
        interaction_types['kurtosis'] = InteractionType(
            name='kurtosis',
            function=self._kurtosis_interaction,
            description='Rolling kurtosis of feature',
            complexity=3,
            vectorbt_optimized=True,
            parameters={'window': 20}
        )
        
        # Momentum interactions
        interaction_types['momentum_divergence'] = InteractionType(
            name='momentum_divergence',
            function=self._momentum_divergence_interaction,
            description='Momentum divergence between features',
            complexity=2,
            vectorbt_optimized=True
        )
        
        interaction_types['momentum_convergence'] = InteractionType(
            name='momentum_convergence',
            function=self._momentum_convergence_interaction,
            description='Momentum convergence between features',
            complexity=2,
            vectorbt_optimized=True
        )
        
        return interaction_types
    
    def generate_interactions(self, 
                            features: pd.DataFrame,
                            targets: Optional[pd.Series] = None) -> List[InteractionResult]:
        """
        Generate interaction features using data-driven approach.
        
        Args:
            features: Input features DataFrame
            targets: Target variable (optional)
            
        Returns:
            List of generated interaction results
        """
        logger.info(f"🚀 Starting data-driven interaction generation")
        logger.info(f"📊 Input features: {features.shape}")
        
        start_time = time.time()
        
        # Analyze data characteristics
        data_characteristics = self._analyze_data_characteristics(features)
        
        # Select optimal interaction types
        selected_types = self._select_interaction_types(data_characteristics)
        
        # Generate feature combinations
        feature_combinations = self._generate_feature_combinations(features.columns.tolist())
        
        # Generate interactions
        interactions = []
        for interaction_type_name in selected_types:
            interaction_type = self.interaction_types[interaction_type_name]
            
            for combo in feature_combinations:
                try:
                    result = self._generate_single_interaction(
                        features, combo, interaction_type, targets
                    )
                    if result:
                        interactions.append(result)
                except Exception as e:
                    logger.warning(f"⚠️ Interaction generation failed: {e}")
                    continue
        
        # Filter and rank interactions
        filtered_interactions = self._filter_interactions(interactions, targets)
        ranked_interactions = self._rank_interactions(filtered_interactions, targets)
        
        # Select top interactions
        selected_interactions = ranked_interactions[:self.max_interactions]
        
        execution_time = time.time() - start_time
        logger.info(f"✅ Generated {len(selected_interactions)} interactions in {execution_time:.2f}s")
        
        return selected_interactions
    
    def _analyze_data_characteristics(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data characteristics to inform interaction selection."""
        characteristics = {}
        
        # Basic statistics
        characteristics['n_features'] = len(features.columns)
        characteristics['n_samples'] = len(features)
        characteristics['data_types'] = features.dtypes.value_counts().to_dict()
        
        # Correlation analysis
        corr_matrix = features.corr()
        characteristics['avg_correlation'] = corr_matrix.abs().mean().mean()
        characteristics['max_correlation'] = corr_matrix.abs().max().max()
        
        # Variance analysis
        characteristics['feature_variance'] = features.var().to_dict()
        characteristics['avg_variance'] = features.var().mean()
        
        # Skewness and kurtosis
        characteristics['feature_skewness'] = features.skew().to_dict()
        characteristics['feature_kurtosis'] = features.kurtosis().to_dict()
        
        # Missing values
        characteristics['missing_values'] = features.isnull().sum().to_dict()
        
        return characteristics
    
    def _select_interaction_types(self, characteristics: Dict[str, Any]) -> List[str]:
        """Select optimal interaction types based on data characteristics."""
        selected_types = []
        
        # Always include basic arithmetic interactions
        selected_types.extend(['product', 'ratio', 'difference', 'sum'])
        
        # Add correlation-based interactions if features are not highly correlated
        if characteristics['avg_correlation'] < 0.7:
            selected_types.extend(['correlation', 'covariance', 'rank_correlation'])
        
        # Add statistical interactions if data has sufficient variance
        if characteristics['avg_variance'] > 0.01:
            selected_types.extend(['skewness', 'kurtosis'])
        
        # Add polynomial interactions for non-normal distributions
        avg_skewness = np.mean([abs(s) for s in characteristics['feature_skewness'].values()])
        if avg_skewness > 0.5:
            selected_types.extend(['quadratic', 'cubic'])
        
        # Add momentum interactions for time series data
        if characteristics['n_samples'] > 50:
            selected_types.extend(['momentum_divergence', 'momentum_convergence'])
        
        # Add z-score interactions for normalization
        selected_types.append('zscore_product')
        
        return list(set(selected_types))  # Remove duplicates
    
    def _generate_feature_combinations(self, feature_names: List[str]) -> List[Tuple[str, ...]]:
        """Generate feature combinations for interactions."""
        combinations_list = []
        
        # Single feature interactions (polynomial, statistical)
        for feature in feature_names:
            combinations_list.append((feature,))
        
        # Two feature interactions
        for combo in combinations(feature_names, 2):
            combinations_list.append(combo)
        
        # Three feature interactions (limited)
        if len(feature_names) <= 10:  # Only for small feature sets
            for combo in combinations(feature_names, 3):
                combinations_list.append(combo)
        
        return combinations_list
    
    def _generate_single_interaction(self, 
                                   features: pd.DataFrame,
                                   feature_combo: Tuple[str, ...],
                                   interaction_type: InteractionType,
                                   targets: Optional[pd.Series]) -> Optional[InteractionResult]:
        """Generate a single interaction feature."""
        try:
            # Extract feature data
            feature_data = [features[feat] for feat in feature_combo]
            
            # Generate interaction
            if len(feature_combo) == 1:
                # Single feature interaction
                result_series = interaction_type.function(feature_data[0])
            else:
                # Multi-feature interaction
                result_series = interaction_type.function(*feature_data)
            
            if result_series is None or result_series.empty:
                return None
            
            # Calculate utility score
            utility_score = self._calculate_utility_score(result_series, targets)
            
            if utility_score < self.utility_threshold:
                return None
            
            # Create feature name
            feature_name = f"{interaction_type.name}_{'_'.join(feature_combo)}"
            
            return InteractionResult(
                feature_name=feature_name,
                feature_series=result_series,
                parent_features=list(feature_combo),
                interaction_type=interaction_type.name,
                utility_score=utility_score,
                metadata={
                    'complexity': interaction_type.complexity,
                    'vectorbt_optimized': interaction_type.vectorbt_optimized,
                    'parameters': interaction_type.parameters
                }
            )
            
        except Exception as e:
            logger.debug(f"⚠️ Single interaction generation failed: {e}")
            return None
    
    def _calculate_utility_score(self, series: pd.Series, targets: Optional[pd.Series]) -> float:
        """Calculate utility score for a feature."""
        try:
            if targets is None:
                # Use variance as utility score
                return float(series.var())
            
            # Calculate correlation with targets
            correlation = series.corr(targets)
            if pd.isna(correlation):
                return 0.0
            
            # Use absolute correlation as utility score
            return abs(correlation)
            
        except Exception as e:
            logger.debug(f"⚠️ Utility score calculation failed: {e}")
            return 0.0
    
    def _filter_interactions(self, 
                           interactions: List[InteractionResult],
                           targets: Optional[pd.Series]) -> List[InteractionResult]:
        """Filter interactions based on quality criteria."""
        filtered = []
        
        for interaction in interactions:
            # Check utility threshold
            if interaction.utility_score < self.utility_threshold:
                continue
            
            # Check for valid values
            if interaction.feature_series.isna().all():
                continue
            
            # Check for infinite values
            if np.isinf(interaction.feature_series).any():
                continue
            
            # Check for constant values
            if interaction.feature_series.nunique() <= 1:
                continue
            
            filtered.append(interaction)
        
        return filtered
    
    def _rank_interactions(self, 
                         interactions: List[InteractionResult],
                         targets: Optional[pd.Series]) -> List[InteractionResult]:
        """Rank interactions by utility score."""
        return sorted(interactions, key=lambda x: x.utility_score, reverse=True)
    
    # Interaction type implementations
    def _product_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Product interaction."""
        return feat1 * feat2
    
    def _ratio_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Ratio interaction."""
        return feat1 / (feat2 + 1e-08)
    
    def _difference_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Difference interaction."""
        return feat1 - feat2
    
    def _sum_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Sum interaction."""
        return feat1 + feat2
    
    def _correlation_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Correlation interaction."""
        window = 20
        if self.enable_vectorbt and VECTORBT_AVAILABLE:
            return rolling_corr(feat1, feat2, window=window)
        else:
            return feat1.rolling(window=window).corr(feat2)
    
    def _covariance_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Covariance interaction."""
        window = 20
        if self.enable_vectorbt and VECTORBT_AVAILABLE:
            return rolling_cov(feat1, feat2, window=window)
        else:
            return feat1.rolling(window=window).cov(feat2)
    
    def _zscore_product_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Z-score product interaction."""
        if self.enable_vectorbt and VECTORBT_AVAILABLE:
            z1 = zscore(feat1)
            z2 = zscore(feat2)
        else:
            z1 = (feat1 - feat1.mean()) / feat1.std()
            z2 = (feat2 - feat2.mean()) / feat2.std()
        
        return z1 * z2
    
    def _rank_correlation_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Rank correlation interaction."""
        window = 20
        if self.enable_vectorbt and VECTORBT_AVAILABLE:
            rank1 = rank(feat1)
            rank2 = rank(feat2)
            return rolling_corr(rank1, rank2, window=window)
        else:
            rank1 = feat1.rank()
            rank2 = feat2.rank()
            return rank1.rolling(window=window).corr(rank2)
    
    def _quadratic_interaction(self, feat: pd.Series) -> pd.Series:
        """Quadratic interaction."""
        return feat ** 2
    
    def _cubic_interaction(self, feat: pd.Series) -> pd.Series:
        """Cubic interaction."""
        return feat ** 3
    
    def _skewness_interaction(self, feat: pd.Series) -> pd.Series:
        """Skewness interaction."""
        window = 20
        if self.enable_vectorbt and VECTORBT_AVAILABLE:
            return rolling_skew(feat, window=window)
        else:
            return feat.rolling(window=window).skew()
    
    def _kurtosis_interaction(self, feat: pd.Series) -> pd.Series:
        """Kurtosis interaction."""
        window = 20
        if self.enable_vectorbt and VECTORBT_AVAILABLE:
            return rolling_kurt(feat, window=window)
        else:
            return feat.rolling(window=window).kurt()
    
    def _momentum_divergence_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Momentum divergence interaction."""
        momentum1 = feat1.pct_change()
        momentum2 = feat2.pct_change()
        return momentum1 - momentum2
    
    def _momentum_convergence_interaction(self, feat1: pd.Series, feat2: pd.Series) -> pd.Series:
        """Momentum convergence interaction."""
        momentum1 = feat1.pct_change()
        momentum2 = feat2.pct_change()
        return momentum1 * momentum2