"""
Negative Learning Feature Generation Module

This module implements the negative learning plugin for Analyst/Tactician tree pipelines.
It discovers failure contexts and generates gated twin features and exception interactions
to improve model performance in challenging market conditions.

Key Components:
1. Failure Context Discovery - Data-driven detection of when features fail
2. Negative Learning Features - Gated twins and exception interactions
3. Model Constraints - Monotone constraints and sample weights
4. Validation Framework - Performance monitoring and SHAP analysis

Time-series safe, fast, and respects latency budgets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

from src.utils.logger import system_logger
from src.utils.math_validation import safe_divide, validate_finite, validate_positive


class FailureContextType(Enum):
    """Types of failure contexts to detect"""
    HIGH_VOLATILITY = "highvol"
    CHOP = "chop"
    WIDE_SPREAD = "widespread"
    OPEN_WINDOW = "open30"
    CLOSE_WINDOW = "last30"
    TRENDING = "trending"
    RANGING = "ranging"


@dataclass
class FailureContext:
    """Represents a detected failure context for a feature"""
    feature_name: str
    context_type: FailureContextType
    threshold: float
    ic_positive: float
    ic_negative: float
    se_positive: float
    se_negative: float
    significance: float
    is_significant: bool


@dataclass
class NegativeLearningConfig:
    """Configuration for negative learning feature generation"""
    # Failure detection thresholds
    ic_significance_threshold: float = 1.5
    r2_chop_threshold: float = 0.3
    volatility_quantile: float = 0.7
    spread_quantile: float = 0.7
    
    # Feature generation
    max_negative_features: int = 5  # Cap gates to 5 max per base feature
    max_gates_per_base_feature: int = 5  # Explicit cap for clarity
    enable_gated_twins: bool = True
    enable_exception_interactions: bool = True
    enable_context_indicators: bool = True
    
    # Model constraints
    enable_monotone_constraints: bool = True
    enable_sample_weights: bool = True
    weight_uncertainty_factor: float = 0.3
    
    # Validation
    stability_selection_bootstrap: int = 80
    stability_selection_threshold: float = 0.6
    min_ic_improvement: float = 0.10


class FailureContextDetector:
    """
    Detects failure contexts where features exhibit sign flips or poor performance.
    Runs once per retrain to identify problematic market conditions.
    """
    
    def __init__(self, config: NegativeLearningConfig):
        self.config = config
        self.logger = system_logger.getChild('FailureContextDetector')
        self.failure_contexts: Dict[str, List[FailureContext]] = {}
        
    def detect_failure_contexts(
        self, 
        features_df: pd.DataFrame, 
        target: pd.Series,
        feature_names: List[str]
    ) -> Dict[str, List[FailureContext]]:
        """
        Discover failure contexts for each feature using data-driven analysis.
        
        Args:
            features_df: Feature matrix
            target: Target variable (returns)
            feature_names: List of features to analyze
            
        Returns:
            Dictionary mapping feature names to their failure contexts
        """
        self.logger.info("🔍 Starting failure context detection...")
        
        # Generate context flags
        context_flags = self._generate_context_flags(features_df)
        
        # Detect failures for each feature
        for feature_name in feature_names:
            if feature_name not in features_df.columns:
                continue
                
            self.logger.debug(f"Analyzing failure contexts for {feature_name}")
            feature_failures = self._analyze_feature_failures(
                features_df[feature_name], 
                target, 
                context_flags,
                feature_name
            )
            
            if feature_failures:
                self.failure_contexts[feature_name] = feature_failures
                self.logger.info(f"Found {len(feature_failures)} failure contexts for {feature_name}")
        
        self.logger.info(f"✅ Failure context detection complete. Found contexts for {len(self.failure_contexts)} features")
        return self.failure_contexts
    
    def _generate_context_flags(self, features_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate soft context flags for different market conditions"""
        context_flags = {}
        
        # High volatility flag (EWMA of volatility, Q70+)
        if 'volatility' in features_df.columns:
            vol_ewma = features_df['volatility'].ewm(span=20).mean()
            vol_threshold = vol_ewma.quantile(self.config.volatility_quantile)
            context_flags['highvol'] = (vol_ewma > vol_threshold).astype(float)
        else:
            # Fallback: use price range volatility
            if 'high' in features_df.columns and 'low' in features_df.columns:
                price_range = features_df['high'] - features_df['low']
                vol_ewma = price_range.ewm(span=20).mean()
                vol_threshold = vol_ewma.quantile(self.config.volatility_quantile)
                context_flags['highvol'] = (vol_ewma > vol_threshold).astype(float)
            else:
                context_flags['highvol'] = pd.Series(0, index=features_df.index)
        
        # Chop flag (low R² of trend fit)
        context_flags['chop'] = self._calculate_chop_flag(features_df)
        
        # Wide spread flag (spread z-score Q70+)
        context_flags['widespread'] = self._calculate_spread_flag(features_df)
        
        # Time-based flags
        context_flags['open30'] = self._calculate_time_flag(features_df, 'open')
        context_flags['last30'] = self._calculate_time_flag(features_df, 'close')
        
        return context_flags
    
    def _calculate_chop_flag(self, features_df: pd.DataFrame) -> pd.Series:
        """Calculate chop flag based on low R² of trend fit"""
        try:
            # Use close price for trend fitting if available
            if 'close' in features_df.columns:
                price = features_df['close']
            elif 'close_price' in features_df.columns:
                price = features_df['close_price']
            else:
                return pd.Series(0, index=features_df.index)
            
            # Calculate rolling R² of linear trend fit
            window = 20
            r2_scores = []
            
            for i in range(len(price)):
                start_idx = max(0, i - window + 1)
                end_idx = i + 1
                
                if end_idx - start_idx < 5:  # Need minimum data points
                    r2_scores.append(0)
                    continue
                
                y = price.iloc[start_idx:end_idx].values
                x = np.arange(len(y)).reshape(-1, 1)
                
                try:
                    reg = LinearRegression().fit(x, y)
                    y_pred = reg.predict(x)
                    r2 = r2_score(y, y_pred)
                    r2_scores.append(max(0, r2))  # Ensure non-negative
                except:
                    r2_scores.append(0)
            
            r2_series = pd.Series(r2_scores, index=price.index)
            # Chop when R² is low (below threshold)
            chop_threshold = self.config.r2_chop_threshold
            return (r2_series < chop_threshold).astype(float)
            
        except Exception as e:
            self.logger.warning(f"Error calculating chop flag: {e}")
            return pd.Series(0, index=features_df.index)
    
    def _calculate_spread_flag(self, features_df: pd.DataFrame) -> pd.Series:
        """Calculate wide spread flag based on spread z-scores"""
        try:
            # Try to find spread-related features
            spread_cols = [col for col in features_df.columns if 'spread' in col.lower()]
            
            if spread_cols:
                spread = features_df[spread_cols[0]]
            elif 'high' in features_df.columns and 'low' in features_df.columns:
                spread = features_df['high'] - features_df['low']
            else:
                return pd.Series(0, index=features_df.index)
            
            # Calculate z-scores
            spread_mean = spread.rolling(50).mean()
            spread_std = spread.rolling(50).std()
            spread_z = safe_divide(spread - spread_mean, spread_std)
            
            # Wide spread when z-score > threshold
            spread_threshold = stats.norm.ppf(self.config.spread_quantile)
            return (spread_z > spread_threshold).astype(float)
            
        except Exception as e:
            self.logger.warning(f"Error calculating spread flag: {e}")
            return pd.Series(0, index=features_df.index)
    
    def _calculate_time_flag(self, features_df: pd.DataFrame, period: str) -> pd.Series:
        """Calculate time-based flags (e.g., first/last 30 minutes)"""
        try:
            if 'timestamp' in features_df.columns:
                timestamps = pd.to_datetime(features_df['timestamp'])
            elif features_df.index.name == 'timestamp' or 'time' in str(features_df.index.name):
                timestamps = pd.to_datetime(features_df.index)
            else:
                return pd.Series(0, index=features_df.index)
            
            # Extract hour and minute
            hours = timestamps.hour
            minutes = timestamps.minute
            
            if period == 'open':
                # First 30 minutes of trading day (assuming 9:00-16:00)
                return ((hours == 9) & (minutes <= 30)).astype(float)
            elif period == 'close':
                # Last 30 minutes of trading day
                return ((hours == 15) & (minutes >= 30)).astype(float)
            else:
                return pd.Series(0, index=features_df.index)
                
        except Exception as e:
            self.logger.warning(f"Error calculating time flag: {e}")
            return pd.Series(0, index=features_df.index)
    
    def _analyze_feature_failures(
        self, 
        feature: pd.Series, 
        target: pd.Series, 
        context_flags: Dict[str, pd.Series],
        feature_name: str
    ) -> List[FailureContext]:
        """Analyze failure contexts for a specific feature"""
        failures = []
        
        for context_name, context_flag in context_flags.items():
            try:
                # Calculate IC in each context bucket
                ic_positive, se_positive = self._calculate_ic_with_bootstrap(
                    feature, target, context_flag > 0.6
                )
                ic_negative, se_negative = self._calculate_ic_with_bootstrap(
                    feature, target, context_flag <= 0.6
                )
                
                # Check for sign flip and significance
                if (np.sign(ic_positive) != np.sign(ic_negative) and
                    abs(ic_positive) / se_positive >= self.config.ic_significance_threshold and
                    abs(ic_negative) / se_negative >= self.config.ic_significance_threshold):
                    
                    context_type = FailureContextType(context_name)
                    significance = min(abs(ic_positive) / se_positive, abs(ic_negative) / se_negative)
                    
                    failure = FailureContext(
                        feature_name=feature_name,
                        context_type=context_type,
                        threshold=0.6,
                        ic_positive=ic_positive,
                        ic_negative=ic_negative,
                        se_positive=se_positive,
                        se_negative=se_negative,
                        significance=significance,
                        is_significant=True
                    )
                    
                    failures.append(failure)
                    self.logger.debug(f"Found failure context: {feature_name} fails in {context_name}")
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing {feature_name} in {context_name}: {e}")
                continue
        
        return failures
    
    def _calculate_ic_with_bootstrap(
        self, 
        feature: pd.Series, 
        target: pd.Series, 
        mask: pd.Series,
        n_bootstrap: int = 100
    ) -> Tuple[float, float]:
        """Calculate IC with block bootstrap standard error"""
        try:
            # Align data
            aligned_data = pd.DataFrame({
                'feature': feature,
                'target': target,
                'mask': mask
            }).dropna()
            
            if len(aligned_data) < 10:
                return 0.0, 1.0
            
            masked_data = aligned_data[aligned_data['mask']]
            
            if len(masked_data) < 5:
                return 0.0, 1.0
            
            # Calculate IC
            ic_values = []
            block_size = max(1, len(masked_data) // 10)  # Block size for bootstrap
            
            for _ in range(n_bootstrap):
                # Block bootstrap
                indices = np.random.choice(
                    len(masked_data) - block_size + 1, 
                    size=len(masked_data) // block_size,
                    replace=True
                )
                
                bootstrap_indices = []
                for idx in indices:
                    bootstrap_indices.extend(range(idx, idx + block_size))
                
                if len(bootstrap_indices) > 0:
                    bootstrap_data = masked_data.iloc[bootstrap_indices[:len(masked_data)]]
                    if len(bootstrap_data) > 1:
                        ic = bootstrap_data['feature'].corr(bootstrap_data['target'])
                        if not np.isnan(ic):
                            ic_values.append(ic)
            
            if len(ic_values) == 0:
                return 0.0, 1.0
            
            ic_mean = np.mean(ic_values)
            ic_se = np.std(ic_values)
            
            return ic_mean, ic_se
            
        except Exception as e:
            self.logger.warning(f"Error calculating IC with bootstrap: {e}")
            return 0.0, 1.0


class NegativeLearningFeatureGenerator:
    """
    Generates negative learning features based on detected failure contexts.
    Creates gated twins, exception interactions, and context indicators.
    """
    
    def __init__(self, config: NegativeLearningConfig):
        self.config = config
        self.logger = system_logger.getChild('NegativeLearningFeatureGenerator')
        self.failure_contexts: Dict[str, List[FailureContext]] = {}
        
    def generate_negative_learning_features(
        self, 
        features_df: pd.DataFrame,
        failure_contexts: Dict[str, List[FailureContext]]
    ) -> pd.DataFrame:
        """
        Generate negative learning features based on failure contexts.
        
        Args:
            features_df: Original feature matrix
            failure_contexts: Detected failure contexts per feature
            
        Returns:
            DataFrame with additional negative learning features
        """
        self.logger.info("🔄 Generating negative learning features...")
        self.failure_contexts = failure_contexts
        
        result_df = features_df.copy()
        negative_features = []
        
        # Generate features for each feature with failure contexts
        for feature_name, contexts in failure_contexts.items():
            if feature_name not in features_df.columns:
                continue
                
            self.logger.debug(f"Generating negative features for {feature_name}")
            
            # Calculate combined failure probability
            p_fail = self._calculate_failure_probability(features_df, contexts)
            
            # Generate all possible gate features first
            all_gate_features = {}
            
            if self.config.enable_gated_twins:
                twin_features = self._generate_gated_twins(
                    features_df[feature_name], p_fail, feature_name
                )
                all_gate_features.update(twin_features.to_dict('series'))
            
            if self.config.enable_exception_interactions:
                interaction_features = self._generate_exception_interactions(
                    features_df[feature_name], p_fail, feature_name
                )
                all_gate_features.update(interaction_features.to_dict('series'))
            
            if self.config.enable_context_indicators:
                context_features = self._generate_context_indicators(
                    features_df, contexts, feature_name
                )
                all_gate_features.update(context_features.to_dict('series'))
            
            # Smart selection: Keep only top 5 most impactful gates
            selected_gates = self._select_top_gates(
                all_gate_features, 
                features_df[feature_name], 
                p_fail,
                max_gates=self.config.max_gates_per_base_feature
            )
            
            # Add selected gates to result
            if selected_gates:
                selected_df = pd.DataFrame(selected_gates, index=features_df.index)
                result_df = pd.concat([result_df, selected_df], axis=1)
                negative_features.extend(selected_df.columns.tolist())
                
                self.logger.debug(f"Selected {len(selected_gates)} gates for {feature_name}: {list(selected_gates.keys())}")
        
        self.logger.info(f"✅ Generated {len(negative_features)} negative learning features")
        return result_df
    
    def _calculate_failure_probability(
        self, 
        features_df: pd.DataFrame, 
        contexts: List[FailureContext]
    ) -> pd.Series:
        """Calculate combined failure probability from all contexts"""
        if not contexts:
            return pd.Series(0, index=features_df.index)
        
        # Generate context flags for this feature's failure contexts
        context_flags = {}
        
        for context in contexts:
            context_name = context.context_type.value
            context_flags[context_name] = self._generate_context_flag(
                features_df, context.context_type
            )
        
        # Combine using soft OR (maximum probability)
        if context_flags:
            p_fail = pd.concat(context_flags.values(), axis=1).max(axis=1)
        else:
            p_fail = pd.Series(0, index=features_df.index)
        
        return p_fail.fillna(0)
    
    def _generate_context_flag(
        self, 
        features_df: pd.DataFrame, 
        context_type: FailureContextType
    ) -> pd.Series:
        """Generate context flag for a specific context type"""
        detector = FailureContextDetector(self.config)
        context_flags = detector._generate_context_flags(features_df)
        return context_flags.get(context_type.value, pd.Series(0, index=features_df.index))
    
    def _generate_gated_twins(
        self, 
        feature: pd.Series, 
        p_fail: pd.Series, 
        feature_name: str
    ) -> pd.DataFrame:
        """Generate gated twin features (positive/negative)"""
        # Gated twin (positive/negative)
        f_pos = feature * (1 - p_fail)  # active where rule should hold
        f_neg = -feature * p_fail       # inverse where it tends to fail
        
        twin_features = pd.DataFrame({
            f"{feature_name}_pos": f_pos,
            f"{feature_name}_neg": f_neg
        }, index=feature.index)
        
        return twin_features
    
    def _generate_exception_interactions(
        self, 
        feature: pd.Series, 
        p_fail: pd.Series, 
        feature_name: str
    ) -> pd.DataFrame:
        """Generate exception interaction features (cheap alternative)"""
        # Exception interaction (cheap alternative)
        f_x_fail = feature * p_fail
        
        interaction_features = pd.DataFrame({
            f"{feature_name}_x_fail": f_x_fail
        }, index=feature.index)
        
        return interaction_features
    
    def _generate_context_indicators(
        self, 
        features_df: pd.DataFrame, 
        contexts: List[FailureContext], 
        feature_name: str
    ) -> pd.DataFrame:
        """Generate context indicator features"""
        context_features = {}
        
        for context in contexts:
            context_name = context.context_type.value
            context_flag = self._generate_context_flag(features_df, context.context_type)
            
            # Include the context flag itself
            context_features[f"{feature_name}_p_{context_name}"] = context_flag
        
        if context_features:
            return pd.DataFrame(context_features, index=features_df.index)
        else:
            return pd.DataFrame(index=features_df.index)


class NegativeLearningValidator:
    """
    Validates negative learning features using bucketed performance and SHAP analysis.
    """
    
    def __init__(self, config: NegativeLearningConfig):
        self.config = config
        self.logger = system_logger.getChild('NegativeLearningValidator')
    
    def validate_negative_learning_features(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        negative_features: List[str],
        failure_contexts: Dict[str, List[FailureContext]]
    ) -> Dict[str, Any]:
        """
        Validate negative learning features using multiple criteria.
        
        Args:
            features_df: Feature matrix including negative learning features
            target: Target variable
            negative_features: List of negative learning feature names
            failure_contexts: Detected failure contexts
            
        Returns:
            Validation results dictionary
        """
        self.logger.info("🔍 Validating negative learning features...")
        
        validation_results = {
            'bucketed_performance': self._validate_bucketed_performance(
                features_df, target, negative_features, failure_contexts
            ),
            'feature_importance': self._validate_feature_importance(
                features_df, target, negative_features
            ),
            'stability_analysis': self._validate_stability(
                features_df, target, negative_features
            )
        }
        
        self.logger.info("✅ Negative learning validation complete")
        return validation_results
    
    def _validate_bucketed_performance(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        negative_features: List[str],
        failure_contexts: Dict[str, List[FailureContext]]
    ) -> Dict[str, Any]:
        """Validate performance within each failure regime"""
        results = {}
        
        for feature_name, contexts in failure_contexts.items():
            if not contexts:
                continue
                
            # Calculate combined failure probability
            generator = NegativeLearningFeatureGenerator(self.config)
            p_fail = generator._calculate_failure_probability(features_df, contexts)
            
            # Bucket by failure probability
            high_fail_mask = p_fail > 0.6
            low_fail_mask = p_fail <= 0.6
            
            # Calculate IC in each bucket
            ic_high_fail = self._calculate_ic(
                features_df[feature_name], target, high_fail_mask
            )
            ic_low_fail = self._calculate_ic(
                features_df[feature_name], target, low_fail_mask
            )
            
            # Check if negative learning features improve performance
            pos_feature = f"{feature_name}_pos"
            neg_feature = f"{feature_name}_neg"
            
            if pos_feature in features_df.columns and neg_feature in features_df.columns:
                ic_pos_high = self._calculate_ic(
                    features_df[pos_feature], target, high_fail_mask
                )
                ic_neg_high = self._calculate_ic(
                    features_df[neg_feature], target, high_fail_mask
                )
                
                results[feature_name] = {
                    'original_ic_high_fail': ic_high_fail,
                    'original_ic_low_fail': ic_low_fail,
                    'pos_ic_high_fail': ic_pos_high,
                    'neg_ic_high_fail': ic_neg_high,
                    'improvement': abs(ic_pos_high) + abs(ic_neg_high) - abs(ic_high_fail)
                }
        
        return results
    
    def _calculate_ic(self, feature: pd.Series, target: pd.Series, mask: pd.Series) -> float:
        """Calculate Information Coefficient"""
        try:
            aligned_data = pd.DataFrame({
                'feature': feature,
                'target': target,
                'mask': mask
            }).dropna()
            
            masked_data = aligned_data[aligned_data['mask']]
            
            if len(masked_data) < 5:
                return 0.0
            
            ic = masked_data['feature'].corr(masked_data['target'])
            return ic if not np.isnan(ic) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating IC: {e}")
            return 0.0
    
    def _validate_feature_importance(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        negative_features: List[str]
    ) -> Dict[str, float]:
        """Validate feature importance using correlation analysis"""
        importance_scores = {}
        
        for feature in negative_features:
            if feature in features_df.columns:
                ic = self._calculate_ic(features_df[feature], target, pd.Series(True, index=features_df.index))
                importance_scores[feature] = abs(ic)
        
        return importance_scores
    
    def _validate_stability(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        negative_features: List[str]
    ) -> Dict[str, Any]:
        """Validate feature stability using rolling correlation"""
        stability_results = {}
        
        for feature in negative_features:
            if feature in features_df.columns:
                # Calculate rolling correlation
                window = 100
                rolling_corr = features_df[feature].rolling(window).corr(target)
                
                stability_results[feature] = {
                    'mean_correlation': rolling_corr.mean(),
                    'correlation_std': rolling_corr.std(),
                    'stability_score': 1 - rolling_corr.std()  # Higher is more stable
                }
        
        return stability_results


class NegativeLearningPlugin:
    """
    Main plugin class that orchestrates the entire negative learning pipeline.
    Integrates with existing Analyst/Tactician pipelines.
    """
    
    def __init__(self, config: Optional[NegativeLearningConfig] = None):
        self.config = config or NegativeLearningConfig()
        self.logger = system_logger.getChild('NegativeLearningPlugin')
        
        # Initialize components
        self.detector = FailureContextDetector(self.config)
        self.generator = NegativeLearningFeatureGenerator(self.config)
        self.validator = NegativeLearningValidator(self.config)
        
        # State
        self.failure_contexts: Dict[str, List[FailureContext]] = {}
        self.negative_features: List[str] = []
        self.validation_results: Dict[str, Any] = {}
    
    def fit(
        self, 
        features_df: pd.DataFrame, 
        target: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> 'NegativeLearningPlugin':
        """
        Fit the negative learning plugin on training data.
        
        Args:
            features_df: Training feature matrix
            target: Training target variable
            feature_names: Optional list of features to analyze
            
        Returns:
            Self for method chaining
        """
        self.logger.info("🎯 Fitting negative learning plugin...")
        
        # Use all numeric features if not specified
        if feature_names is None:
            feature_names = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Detect failure contexts
        self.failure_contexts = self.detector.detect_failure_contexts(
            features_df, target, feature_names
        )
        
        self.logger.info(f"✅ Plugin fitted with {len(self.failure_contexts)} features having failure contexts")
        return self
    
    def transform(
        self, 
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform features by adding negative learning features.
        
        Args:
            features_df: Feature matrix to transform
            
        Returns:
            Transformed feature matrix with negative learning features
        """
        self.logger.info("🔄 Transforming features with negative learning...")
        
        # Generate negative learning features
        transformed_df = self.generator.generate_negative_learning_features(
            features_df, self.failure_contexts
        )
        
        # Update negative features list
        new_features = [col for col in transformed_df.columns if col not in features_df.columns]
        self.negative_features.extend(new_features)
        
        self.logger.info(f"✅ Transformed features. Added {len(new_features)} negative learning features")
        return transformed_df
    
    def fit_transform(
        self, 
        features_df: pd.DataFrame, 
        target: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            features_df: Feature matrix
            target: Target variable
            feature_names: Optional list of features to analyze
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(features_df, target, feature_names).transform(features_df)
    
    def validate(
        self, 
        features_df: pd.DataFrame, 
        target: pd.Series
    ) -> Dict[str, Any]:
        """
        Validate the negative learning features.
        
        Args:
            features_df: Feature matrix including negative learning features
            target: Target variable
            
        Returns:
            Validation results
        """
        self.logger.info("🔍 Validating negative learning features...")
        
        self.validation_results = self.validator.validate_negative_learning_features(
            features_df, target, self.negative_features, self.failure_contexts
        )
        
        return self.validation_results
    
    def get_monotone_constraints(self, feature_names: List[str]) -> List[int]:
        """
        Get monotone constraints for tree-based models.
        
        Args:
            feature_names: List of feature names in model order
            
        Returns:
            List of monotone constraints (-1, 0, 1)
        """
        constraints = []
        
        for feature_name in feature_names:
            if feature_name.endswith('_pos'):
                # Positive features should have positive monotonicity
                constraints.append(1)
            elif feature_name.endswith('_neg'):
                # Negative features should have negative monotonicity
                constraints.append(-1)
            else:
                # No constraint for other features
                constraints.append(0)
        
        return constraints
    
    def get_sample_weights(
        self, 
        features_df: pd.DataFrame, 
        base_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Get sample weights that down-weight uncertain failure zones.
        
        Args:
            features_df: Feature matrix
            base_weights: Optional base sample weights
            
        Returns:
            Sample weights
        """
        if base_weights is None:
            base_weights = pd.Series(1.0, index=features_df.index)
        
        # Calculate maximum failure probability across all features
        p_fail_max = pd.Series(0, index=features_df.index)
        
        for feature_name, contexts in self.failure_contexts.items():
            if contexts:
                generator = NegativeLearningFeatureGenerator(self.config)
                p_fail = generator._calculate_failure_probability(features_df, contexts)
                p_fail_max = np.maximum(p_fail_max, p_fail)
        
        # Apply uncertainty weighting
        uncertainty_factor = self.config.weight_uncertainty_factor
        weights = base_weights * (0.7 + 0.3 * (1 - p_fail_max))
        
        return weights
    
    def _select_top_gates(
        self, 
        all_gate_features: Dict[str, pd.Series], 
        base_feature: pd.Series,
        p_fail: pd.Series,
        max_gates: int = 5
    ) -> Dict[str, pd.Series]:
        """
        Select top N most impactful gate features based on multiple criteria.
        
        Args:
            all_gate_features: All generated gate features
            base_feature: Original base feature
            p_fail: Failure probability
            max_gates: Maximum number of gates to select
            
        Returns:
            Selected gate features
        """
        if len(all_gate_features) <= max_gates:
            return all_gate_features
        
        # Calculate impact scores for each gate
        gate_scores = {}
        
        for gate_name, gate_series in all_gate_features.items():
            try:
                # Calculate multiple impact metrics
                ic_score = self._calculate_gate_ic_score(gate_series, base_feature)
                stability_score = self._calculate_gate_stability(gate_series)
                uniqueness_score = self._calculate_gate_uniqueness(gate_series, all_gate_features)
                context_score = self._calculate_gate_context_score(gate_name, p_fail)
                
                # Weighted composite score
                composite_score = (
                    0.4 * ic_score +           # IC improvement
                    0.3 * stability_score +    # Stability
                    0.2 * uniqueness_score +   # Uniqueness
                    0.1 * context_score        # Context relevance
                )
                
                gate_scores[gate_name] = composite_score
                
            except Exception as e:
                self.logger.warning(f"Error scoring gate {gate_name}: {e}")
                gate_scores[gate_name] = 0.0
        
        # Select top N gates by score
        sorted_gates = sorted(gate_scores.items(), key=lambda x: x[1], reverse=True)
        selected_gates = dict(sorted_gates[:max_gates])
        
        self.logger.debug(f"Gate selection scores: {dict(sorted_gates)}")
        
        return {name: all_gate_features[name] for name in selected_gates.keys()}
    
    def _calculate_gate_ic_score(self, gate_series: pd.Series, base_feature: pd.Series) -> float:
        """Calculate IC improvement score for a gate feature"""
        try:
            # Calculate IC for gate vs base feature
            gate_ic = abs(gate_series.corr(base_feature))
            base_ic = abs(base_feature.corr(base_feature))  # Should be 1.0
            
            # IC improvement (higher is better)
            ic_improvement = max(0, gate_ic - base_ic)
            return min(1.0, ic_improvement * 10)  # Scale to 0-1
            
        except Exception:
            return 0.0
    
    def _calculate_gate_stability(self, gate_series: pd.Series) -> float:
        """Calculate stability score for a gate feature"""
        try:
            # Rolling correlation stability
            window = min(100, len(gate_series) // 4)
            if window < 10:
                return 0.5
            
            rolling_corr = gate_series.rolling(window).corr(gate_series.shift(1))
            stability = 1.0 - rolling_corr.std()
            return max(0.0, min(1.0, stability))
            
        except Exception:
            return 0.5
    
    def _calculate_gate_uniqueness(self, gate_series: pd.Series, all_gates: Dict[str, pd.Series]) -> float:
        """Calculate uniqueness score (low correlation with other gates)"""
        try:
            if len(all_gates) <= 1:
                return 1.0
            
            correlations = []
            for other_name, other_series in all_gates.items():
                if other_name != gate_series.name:
                    corr = abs(gate_series.corr(other_series))
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if not correlations:
                return 1.0
            
            # Uniqueness = 1 - average correlation with other gates
            avg_corr = np.mean(correlations)
            return max(0.0, 1.0 - avg_corr)
            
        except Exception:
            return 0.5
    
    def _calculate_gate_context_score(self, gate_name: str, p_fail: pd.Series) -> float:
        """Calculate context relevance score based on gate name and failure probability"""
        try:
            # Higher score for gates that align with high failure probability
            if 'pos' in gate_name or 'positive' in gate_name:
                # Positive gates should be active when failure prob is low
                context_score = 1.0 - p_fail.mean()
            elif 'neg' in gate_name or 'negative' in gate_name:
                # Negative gates should be active when failure prob is high
                context_score = p_fail.mean()
            elif 'fail' in gate_name or 'exception' in gate_name:
                # Exception gates should align with failure probability
                context_score = p_fail.mean()
            else:
                # Context indicators - check if they're meaningful
                context_score = 0.5 if p_fail.std() > 0.1 else 0.2
            
            return max(0.0, min(1.0, context_score))
            
        except Exception:
            return 0.5
    
    def get_feature_importance_scores(self) -> Dict[str, float]:
        """Get feature importance scores for negative learning features"""
        if not self.validation_results:
            return {}
        
        return self.validation_results.get('feature_importance', {})
    
    def get_failure_contexts(self) -> Dict[str, List[FailureContext]]:
        """Get detected failure contexts"""
        return self.failure_contexts
    
    def get_negative_features(self) -> List[str]:
        """Get list of generated negative learning features"""
        return self.negative_features