"""
Specialist Causal Validator - CausalForestDML Validation Layer

Implements causal validation for specialist models using CausalForestDML:
- Causal effect estimation for specialist signals
- Regime-dependent heterogeneity analysis
- Specialist combination causal optimization
- False signal reduction through causal validation

Key Components:
1. CausalSpecialistValidator - Main validation class
2. SpecialistTreatmentBuilder - Treatment variable construction
3. CausalEffectAnalyzer - Effect analysis and reporting
4. SpecialistCombinationOptimizer - Multi-specialist causal optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import warnings
from pathlib import Path
from scipy.stats import norm

# EconML imports for causal inference
try:
    from econml.dml import CausalForestDML
    from econml.dr import DRLearner
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    CausalForestDML = None
    DRLearner = None
    warnings.warn("EconML not available - causal validation will be disabled")

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb

# Project imports
from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from src.training.steps.labeling.causal_targets import CausalTargetComputer
from .specialist_data_standard import SpecialistType
from .specialist_interface import SpecialistDataInterface

logger = logging.getLogger(__name__)


@dataclass
class CausalValidationMetrics:
    """Container for causal validation results."""
    specialist_name: str
    cate_mean: float
    cate_se: float
    cate_median: float
    causal_significance: float  # Proportion of significant effects
    causal_lift: float  # Improvement over correlation
    regime_heterogeneity: float  # Effect variation across regimes
    confidence_interval: Tuple[float, float]
    n_samples: int
    validation_timestamp: datetime
    treatment_type: str
    effect_size_classification: str  # 'small', 'medium', 'large'
    false_signal_probability: float


@dataclass
class SpecialistTreatmentConfig:
    """Configuration for treatment variable construction."""
    treatment_type: str = 'binary'  # 'binary', 'continuous', 'categorical'
    threshold_method: str = 'percentile'  # 'percentile', 'std', 'fixed'
    threshold_value: float = 0.5  # For percentile method
    min_treatment_frequency: float = 0.1  # Minimum treatment frequency
    max_treatment_frequency: float = 0.9  # Maximum treatment frequency


@dataclass
class CausalValidationConfig:
    """Configuration for causal validation."""
    n_estimators: int = 500
    max_samples: float = 0.5
    min_samples_leaf: int = 50
    inference_method: str = 'blp'  # 'blp', 'bootstrap'
    n_jobs: int = 2
    random_state: int = 42
    significance_level: float = 0.05
    heterogeneity_features: List[str] = field(default_factory=list)
    confounder_features: List[str] = field(default_factory=list)


class SpecialistTreatmentBuilder:
    """Builds treatment variables from specialist signals for causal analysis."""
    
    def __init__(self, config: SpecialistTreatmentConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def build_treatment(self, specialist_outputs: pd.DataFrame, 
                       specialist_name: str) -> pd.Series:
        """
        Convert specialist signals to causal treatment variables.
        
        Args:
            specialist_outputs: DataFrame containing specialist predictions
            specialist_name: Name of the specialist
            
        Returns:
            Treatment variable as pandas Series
        """
        try:
            # Extract specialist signal
            signal = self._extract_specialist_signal(specialist_outputs, specialist_name)
            
            if self.config.treatment_type == 'binary':
                treatment = self._build_binary_treatment(signal)
            elif self.config.treatment_type == 'continuous':
                treatment = self._build_continuous_treatment(signal)
            elif self.config.treatment_type == 'categorical':
                treatment = self._build_categorical_treatment(signal)
            else:
                raise ValueError(f"Unknown treatment type: {self.config.treatment_type}")
            
            # Validate treatment frequency
            treatment = self._validate_treatment_frequency(treatment)
            
            self.logger.info(f"Built {self.config.treatment_type} treatment for {specialist_name}")
            return treatment
            
        except Exception as e:
            self.logger.error(f"Failed to build treatment for {specialist_name}: {e}")
            # Return fallback treatment
            return pd.Series(0, index=specialist_outputs.index, name='treatment')
    
    def _extract_specialist_signal(self, specialist_outputs: pd.DataFrame, 
                                 specialist_name: str) -> pd.Series:
        """Extract the primary signal from specialist outputs."""
        # Try standard prediction columns
        signal_candidates = [
            f'{specialist_name}_prediction',
            f'{specialist_name}_probability',
            f'{specialist_name}_score',
            'specialist_prediction',
            'specialist_probability',
            'prediction',
            'probability',
            'score'
        ]
        
        for candidate in signal_candidates:
            if candidate in specialist_outputs.columns:
                return specialist_outputs[candidate]
        
        # Fallback to first numeric column
        numeric_cols = specialist_outputs.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return specialist_outputs[numeric_cols[0]]
        
        raise ValueError(f"No suitable signal found for specialist {specialist_name}")
    
    def _build_binary_treatment(self, signal: pd.Series) -> pd.Series:
        """Build binary treatment variable."""
        if self.config.threshold_method == 'percentile':
            threshold = signal.quantile(self.config.threshold_value)
        elif self.config.threshold_method == 'std':
            threshold = signal.mean() + signal.std() * self.config.threshold_value
        elif self.config.threshold_method == 'fixed':
            threshold = self.config.threshold_value
        else:
            threshold = signal.median()
        
        treatment = (signal > threshold).astype(int)
        return treatment.rename('treatment')
    
    def _build_continuous_treatment(self, signal: pd.Series) -> pd.Series:
        """Build continuous treatment variable."""
        # Standardize the signal
        treatment = (signal - signal.mean()) / (signal.std() + 1e-8)
        return treatment.rename('treatment')
    
    def _build_categorical_treatment(self, signal: pd.Series) -> pd.Series:
        """Build categorical treatment variable."""
        # Discretize into quartiles
        treatment = pd.qcut(signal, q=4, labels=False, duplicates='drop')
        return treatment.rename('treatment')
    
    def _validate_treatment_frequency(self, treatment: pd.Series) -> pd.Series:
        """Validate treatment frequency constraints."""
        treatment_freq = treatment.mean()
        
        if treatment_freq < self.config.min_treatment_frequency:
            self.logger.warning(f"Treatment frequency {treatment_freq:.3f} below minimum")
            # Adjust threshold to meet minimum frequency
            if treatment.nunique() > 1:
                threshold = np.percentile(treatment, (1 - self.config.min_treatment_frequency) * 100)
                treatment = (treatment > threshold).astype(int)
        
        elif treatment_freq > self.config.max_treatment_frequency:
            self.logger.warning(f"Treatment frequency {treatment_freq:.3f} above maximum")
            # Adjust threshold to meet maximum frequency
            if treatment.nunique() > 1:
                threshold = np.percentile(treatment, self.config.max_treatment_frequency * 100)
                treatment = (treatment > threshold).astype(int)
        
        return treatment


class CausalEffectAnalyzer:
    """Analyzes and reports causal effects from specialist validation."""
    
    def __init__(self, config: CausalValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_causal_effects(self, cate_values: np.ndarray, 
                               se_values: np.ndarray,
                               treatment: pd.Series,
                               outcome: pd.Series,
                               effect_modifiers: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze causal effects and compute validation metrics.
        
        Args:
            cate_values: Conditional Average Treatment Effects
            se_values: Standard errors for CATE
            treatment: Treatment variable
            outcome: Outcome variable
            effect_modifiers: Heterogeneity features
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {}
        
        # Basic effect statistics
        analysis['cate_mean'] = np.mean(cate_values)
        analysis['cate_median'] = np.median(cate_values)
        analysis['cate_std'] = np.std(cate_values)
        analysis['cate_se_mean'] = np.mean(se_values)
        
        # Significance testing
        z_scores = cate_values / (se_values + 1e-8)
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))  # Two-sided test
        analysis['causal_significance'] = np.mean(p_values < self.config.significance_level)
        
        # Effect size classification
        analysis['effect_size_classification'] = self._classify_effect_size(analysis['cate_mean'])
        
        # Confidence interval
        analysis['confidence_interval'] = (
            analysis['cate_mean'] - 1.96 * analysis['cate_se_mean'],
            analysis['cate_mean'] + 1.96 * analysis['cate_se_mean']
        )
        
        # Heterogeneity analysis
        if effect_modifiers is not None:
            analysis['regime_heterogeneity'] = self._analyze_heterogeneity(
                cate_values, effect_modifiers
            )
        
        # Causal lift vs correlation
        analysis['causal_lift'] = self._compute_causal_lift(cate_values, treatment, outcome)
        
        # False signal probability
        analysis['false_signal_probability'] = self._estimate_false_signal_probability(
            cate_values, se_values, p_values
        )
        
        return analysis
    
    def _classify_effect_size(self, cate_mean: float) -> str:
        """Classify effect size based on Cohen's conventions."""
        abs_effect = abs(cate_mean)
        if abs_effect < 0.01:
            return 'small'
        elif abs_effect < 0.03:
            return 'medium'
        else:
            return 'large'
    
    def _analyze_heterogeneity(self, cate_values: np.ndarray, 
                              effect_modifiers: pd.DataFrame) -> float:
        """Analyze heterogeneity of treatment effects."""
        # Simple heterogeneity measure: variance of CATE across different regimes
        heterogeneity_scores = []
        
        for col in effect_modifiers.columns:
            if effect_modifiers[col].nunique() > 1:
                # Group by different values of the effect modifier
                groups = effect_modifiers[col].values
                unique_groups = np.unique(groups[~pd.isna(groups)])
                
                if len(unique_groups) > 1:
                    group_cates = []
                    for group in unique_groups:
                        mask = groups == group
                        if np.sum(mask) > 0:
                            group_cates.append(np.mean(cate_values[mask]))
                    
                    if len(group_cates) > 1:
                        # Coefficient of variation across groups
                        group_mean = np.mean(group_cates)
                        group_std = np.std(group_cates)
                        heterogeneity_scores.append(group_std / (abs(group_mean) + 1e-8))
        
        return np.mean(heterogeneity_scores) if heterogeneity_scores else 0.0
    
    def _compute_causal_lift(self, cate_values: np.ndarray, 
                           treatment: pd.Series, outcome: pd.Series) -> float:
        """Compute causal lift compared to simple correlation."""
        # Simple correlation between treatment and outcome
        correlation = np.corrcoef(treatment.values, outcome.values)[0, 1]
        
        # Mean causal effect
        mean_causal_effect = np.mean(cate_values)
        
        # Lift ratio (causal effect vs correlation)
        if abs(correlation) > 1e-8:
            lift = abs(mean_causal_effect) / abs(correlation)
        else:
            lift = abs(mean_causal_effect)
        
        return lift
    
    def _estimate_false_signal_probability(self, cate_values: np.ndarray,
                                         se_values: np.ndarray,
                                         p_values: np.ndarray) -> float:
        """Estimate probability that signal is false (non-causal)."""
        # Proportion of effects that are not statistically significant
        non_significant = np.mean(p_values >= self.config.significance_level)
        
        # Adjust for effect size - smaller effects more likely to be false
        small_effects = np.mean(np.abs(cate_values) < 0.01)
        
        # Combine significance and effect size
        false_signal_prob = 0.7 * non_significant + 0.3 * small_effects
        
        return false_signal_prob


class CausalSpecialistValidator:
    """
    Main causal validation class for specialist models.
    
    Uses CausalForestDML to validate specialist signals, distinguishing
    correlation from causation and providing regime-dependent analysis.
    """
    
    def __init__(self, config: Optional[CausalValidationConfig] = None,
                 treatment_config: Optional[SpecialistTreatmentConfig] = None):
        """
        Initialize the causal validator.
        
        Args:
            config: Causal validation configuration
            treatment_config: Treatment variable configuration
        """
        self.config = config or CausalValidationConfig()
        self.treatment_config = treatment_config or SpecialistTreatmentConfig()
        self.treatment_builder = SpecialistTreatmentBuilder(self.treatment_config)
        self.effect_analyzer = CausalEffectAnalyzer(self.config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize causal computer if available
        if ECONML_AVAILABLE:
            self.causal_computer = CausalTargetComputer(verbose=True)
        else:
            self.causal_computer = None
            self.logger.warning("EconML not available - causal validation disabled")
        
        # Cache for validation results
        self._validation_cache = {}
    
    def validate_specialist(self, specialist_outputs: pd.DataFrame,
                           market_data: pd.DataFrame,
                           regime_labels: pd.Series,
                           specialist_name: str) -> CausalValidationMetrics:
        """
        Validate a specialist model using causal inference.
        
        Args:
            specialist_outputs: DataFrame containing specialist predictions
            market_data: DataFrame containing market variables
            regime_labels: Series containing regime transition labels
            specialist_name: Name of the specialist
            
        Returns:
            CausalValidationMetrics containing validation results
        """
        if not ECONML_AVAILABLE:
            raise ImportError("EconML is required for causal validation")
        
        # Check cache
        cache_key = f"{specialist_name}_{hash((specialist_outputs.shape, market_data.shape))}"
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]
        
        try:
            self.logger.info(f"Starting causal validation for {specialist_name}")
            
            # 1. Build treatment variable
            treatment = self.treatment_builder.build_treatment(specialist_outputs, specialist_name)
            
            # 2. Extract confounders
            confounders = self._extract_confounders(market_data)
            
            # 3. Create heterogeneity features
            effect_modifiers = self._create_heterogeneity_features(market_data)
            
            # 4. Align all data
            aligned_data = self._align_data(treatment, regime_labels, confounders, effect_modifiers)
            
            # 5. Estimate causal effects using CausalForestDML
            causal_results = self._estimate_causal_effects(
                aligned_data['treatment'],
                aligned_data['outcome'],
                aligned_data['confounders'],
                aligned_data['effect_modifiers']
            )
            
            # 6. Analyze effects
            analysis = self.effect_analyzer.analyze_causal_effects(
                causal_results['cate'],
                causal_results['se'],
                aligned_data['treatment'],
                aligned_data['outcome'],
                aligned_data['effect_modifiers']
            )
            
            # 7. Create metrics object
            metrics = CausalValidationMetrics(
                specialist_name=specialist_name,
                cate_mean=analysis['cate_mean'],
                cate_se=analysis['cate_se_mean'],
                cate_median=analysis['cate_median'],
                causal_significance=analysis['causal_significance'],
                causal_lift=analysis['causal_lift'],
                regime_heterogeneity=analysis.get('regime_heterogeneity', 0.0),
                confidence_interval=analysis['confidence_interval'],
                n_samples=len(aligned_data['treatment']),
                validation_timestamp=datetime.now(),
                treatment_type=self.treatment_config.treatment_type,
                effect_size_classification=analysis['effect_size_classification'],
                false_signal_probability=analysis['false_signal_probability']
            )
            
            # Cache results
            self._validation_cache[cache_key] = metrics
            
            self.logger.info(f"Causal validation completed for {specialist_name}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Causal validation failed for {specialist_name}: {e}")
            # Return default metrics
            return CausalValidationMetrics(
                specialist_name=specialist_name,
                cate_mean=0.0,
                cate_se=1.0,
                cate_median=0.0,
                causal_significance=0.0,
                causal_lift=0.0,
                regime_heterogeneity=0.0,
                confidence_interval=(0.0, 0.0),
                n_samples=0,
                validation_timestamp=datetime.now(),
                treatment_type=self.treatment_config.treatment_type,
                effect_size_classification='small',
                false_signal_probability=1.0
            )
    
    def _extract_confounders(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Extract market state variables as confounders."""
        confounders = pd.DataFrame(index=market_data.index)
        
        # Standard confounder features
        confounder_candidates = [
            'volatility', 'volume', 'spread', 'momentum', 'trend',
            'rsi', 'macd', 'bollinger_position', 'atr', 'vwap'
        ]
        
        for candidate in confounder_candidates:
            matching_cols = [col for col in market_data.columns if candidate in col.lower()]
            if matching_cols:
                confounders[candidate] = market_data[matching_cols[0]]
        
        # Add computed confounders if not present
        if 'volatility' not in confounders.columns and 'close' in market_data.columns:
            confounders['volatility'] = market_data['close'].pct_change().rolling(20).std()
        
        if 'volume' not in confounders.columns and 'volume' in market_data.columns:
            confounders['volume'] = market_data['volume']
        
        if 'spread' not in confounders.columns and 'high' in market_data.columns and 'low' in market_data.columns:
            confounders['spread'] = (market_data['high'] - market_data['low']) / market_data['close']
        
        return confounders.fillna(method='ffill').fillna(method='bfill')
    
    def _create_heterogeneity_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create effect modifiers for regime-dependent analysis."""
        effect_modifiers = pd.DataFrame(index=market_data.index)
        
        # Time-based features
        if isinstance(market_data.index, pd.DatetimeIndex):
            effect_modifiers['hour_of_day'] = market_data.index.hour
            effect_modifiers['day_of_week'] = market_data.index.dayofweek
            effect_modifiers['is_us_session'] = ((market_data.index.hour >= 13) & 
                                                 (market_data.index.hour <= 22)).astype(int)
        
        # Market regime features
        if 'volatility' in market_data.columns:
            vol = market_data['volatility'].fillna(0)
            effect_modifiers['volatility_regime'] = pd.qcut(vol, q=3, labels=False, duplicates='drop')
            effect_modifiers['high_volatility'] = (vol > vol.quantile(0.8)).astype(int)
        
        # Trend features
        if 'close' in market_data.columns:
            returns = market_data['close'].pct_change()
            effect_modifiers['trend_strength'] = returns.rolling(20).mean()
            effect_modifiers['is_uptrend'] = (effect_modifiers['trend_strength'] > 0).astype(int)
        
        # Volume features
        if 'volume' in market_data.columns:
            volume = market_data['volume'].fillna(0)
            effect_modifiers['volume_regime'] = pd.qcut(volume, q=3, labels=False, duplicates='drop')
            effect_modifiers['high_volume'] = (volume > volume.quantile(0.8)).astype(int)
        
        return effect_modifiers.fillna(0)
    
    def _align_data(self, treatment: pd.Series, outcome: pd.Series,
                    confounders: pd.DataFrame, effect_modifiers: pd.DataFrame) -> Dict[str, Any]:
        """Align all data to common index."""
        # Find common index
        common_index = treatment.index.intersection(outcome.index)
        common_index = common_index.intersection(confounders.index)
        common_index = common_index.intersection(effect_modifiers.index)
        
        # Drop NaN values
        valid_mask = (
            treatment.loc[common_index].notna() &
            outcome.loc[common_index].notna() &
            confounders.loc[common_index].notna().all(axis=1) &
            effect_modifiers.loc[common_index].notna().all(axis=1)
        )
        
        final_index = common_index[valid_mask]
        
        return {
            'treatment': treatment.loc[final_index],
            'outcome': outcome.loc[final_index],
            'confounders': confounders.loc[final_index],
            'effect_modifiers': effect_modifiers.loc[final_index]
        }
    
    def _estimate_causal_effects(self, treatment: pd.Series, outcome: pd.Series,
                                confounders: pd.DataFrame, effect_modifiers: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Estimate causal effects using CausalForestDML."""
        if self.causal_computer is None:
            raise ImportError("CausalTargetComputer not available")
        
        # Convert to numpy arrays
        T = treatment.values
        Y = outcome.values
        X = effect_modifiers.values  # Effect modifiers
        W = confounders.values  # Confounders
        
        # Use the causal computer to estimate effects
        try:
            results_df = self.causal_computer.compute_orf_causal_metrics(
                X=pd.DataFrame(X, index=effect_modifiers.index, columns=effect_modifiers.columns),
                Y=pd.Series(Y, index=effect_modifiers.index),
                T=pd.Series(T, index=effect_modifiers.index),
                W=pd.DataFrame(W, index=confounders.index, columns=confounders.columns),
                n_trees=self.config.n_estimators,
                min_leaf_size=self.config.min_samples_leaf,
                max_samples=self.config.max_samples,
                use_fast_approximation=True  # Use CausalForestDML
            )
            
            return {
                'cate': results_df['cate'].values,
                'se': results_df['se'].values,
                'p_values': results_df['p_value'].values
            }
            
        except Exception as e:
            self.logger.error(f"Causal effect estimation failed: {e}")
            # Return default values
            n_samples = len(T)
            return {
                'cate': np.zeros(n_samples),
                'se': np.ones(n_samples),
                'p_values': np.ones(n_samples)
            }
    
    def get_validation_summary(self, validation_results: List[CausalValidationMetrics]) -> Dict[str, Any]:
        """Generate summary of validation results across multiple specialists."""
        if not validation_results:
            return {}
        
        summary = {
            'total_specialists': len(validation_results),
            'validation_timestamp': datetime.now(),
            'specialist_rankings': [],
            'aggregate_metrics': {}
        }
        
        # Rank specialists by causal validation score
        specialist_scores = []
        for result in validation_results:
            # Composite score: significance * lift * (1 - false_signal_prob)
            score = (result.causal_significance * 
                    result.causal_lift * 
                    (1 - result.false_signal_probability))
            specialist_scores.append({
                'specialist_name': result.specialist_name,
                'validation_score': score,
                'cate_mean': result.cate_mean,
                'causal_significance': result.causal_significance,
                'causal_lift': result.causal_lift,
                'false_signal_probability': result.false_signal_probability
            })
        
        summary['specialist_rankings'] = sorted(
            specialist_scores, 
            key=lambda x: x['validation_score'], 
            reverse=True
        )
        
        # Aggregate metrics
        summary['aggregate_metrics'] = {
            'mean_causal_significance': np.mean([r.causal_significance for r in validation_results]),
            'mean_causal_lift': np.mean([r.causal_lift for r in validation_results]),
            'mean_false_signal_probability': np.mean([r.false_signal_probability for r in validation_results]),
            'significant_specialists': sum(1 for r in validation_results if r.causal_significance > 0.5),
            'high_lift_specialists': sum(1 for r in validation_results if r.causal_lift > 1.5)
        }
        
        return summary
