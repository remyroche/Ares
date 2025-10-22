"""
Label Quality Scoring and Optimization

This module implements comprehensive label quality assessment to ensure labels are
learnable by ML models and generalize well.

Key Features:
- Predictability assessment using baseline models
- Stability measurement across rolling folds
- Consistency evaluation via mutual information
- Balance assessment for class distribution
- SNR proxy calculation using information coefficient
- Composite Label Quality Score (LQS) calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_data_preview, tprint_data_format
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy.stats import entropy
import warnings
from abc import ABC

# Import BaseStep
from src.training.steps.base_step import BaseStep

# Import matrix operations for vectorized computations
try:
    from src.utils.matrix_operations import UnifiedMatrixOperations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

# Note: tprint and hardware utilities are available through BaseStep
# No need for direct imports as they're inherited from BaseStep

# Import ML optimization utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    BAYESIAN_OPTIMIZER_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZER_AVAILABLE = False
    tprint_warning("Bayesian TPE optimizer not available, will use grid search if needed")

try:
    from src.utils.ml_common.optimization.pareto import ParetoOptimizer, ParetoFront, Solution
    PARETO_OPTIMIZER_AVAILABLE = True
except ImportError:
    PARETO_OPTIMIZER_AVAILABLE = False
    tprint_warning("Pareto optimizer not available, will fallback to simple ranking")

# Import cross-validation utilities
try:
    from src.utils.ml_common.validation.cross_validation import CrossValidator
    CV_UTILITIES_AVAILABLE = True
except ImportError:
    CV_UTILITIES_AVAILABLE = False
    tprint_warning("CV utilities not available")

# Import OOF stacking utilities
try:
    from src.utils.ml_common.ensembles.oof_stacking_ensemble_manager import OOFStackingEnsembleManager
    OOF_AVAILABLE = True
except ImportError:
    OOF_AVAILABLE = False
    tprint_warning("OOF stacking not available")


class QualityMetric(Enum):
    """Enumeration of quality metrics."""
    PREDICTABILITY = "predictability"  # AUC/PR-AUC from baselines
    STABILITY = "stability"  # Variance of AUC across folds, PSI
    CONSISTENCY = "consistency"  # Mutual information between labels
    BALANCE = "balance"  # Class balance
    SNR_PROXY = "snr_proxy"  # |IC| between features and labels
    SPARSITY = "sparsity"  # Label sparsity (minimum positive class ratio)


@dataclass
class QualityScoringConfig:
    """Configuration for quality scoring."""
    
    # Baseline model settings
    baseline_models: List[str] = field(default_factory=lambda: ['logistic', 'random_forest'])
    test_size: float = 0.2  # Test set size
    n_splits: int = 5  # Number of CV splits
    random_state: int = 42
    
    # Feature engineering
    enable_feature_engineering: bool = True
    feature_window: int = 20  # Window for feature calculation
    n_features: int = 10  # Number of features to generate
    
    # Quality thresholds (DEPRECATED - not used in scoring, kept for reporting only)
    min_auc_threshold: float = 0.55
    max_auc_std_threshold: float = 0.03
    min_psi_threshold: float = 0.1
    max_flip_rate_threshold: float = 0.15
    min_balance_threshold: float = 0.35
    max_balance_threshold: float = 0.65
    min_sparsity_threshold: float = 0.05  # Minimum 5% positive class
    
    # LQS calculation
    lqs_components: List[str] = field(default_factory=lambda: [
        'predictability', 'stability', 'consistency', 'balance', 'snr_proxy', 'sparsity'
    ])
    
    # Deprecated: lqs_weights kept for backward compatibility but not used
    lqs_weights: Dict[str, float] = field(default_factory=lambda: {
        'predictability': 0.25,
        'stability': 0.2,
        'consistency': 0.2,
        'balance': 0.2,
        'snr_proxy': 0.1,
        'sparsity': 0.05
    })
    
    # Pareto optimization
    enable_pareto_optimization: bool = True
    pareto_objectives: List[str] = field(default_factory=lambda: [
        'predictability', 'stability', 'consistency', 'balance', 'sparsity'
    ])
    
    # Optimization settings
    enable_optimization: bool = True
    optimization_method: str = 'bayesian'  # 'bayesian' or 'grid'
    n_trials: int = 100
    optimization_metric: str = 'lqs'  # 'lqs' or 'auc'
    
    # Quality checks
    min_samples_for_evaluation: int = 100
    max_evaluation_time_seconds: int = 300


@dataclass
class QualityMetrics:
    """Container for quality metrics."""
    
    # Core metrics
    predictability: float = 0.0
    stability: float = 0.0
    consistency: float = 0.0
    balance: float = 0.0
    snr_proxy: float = 0.0
    sparsity: float = 0.0
    
    # Composite score
    lqs_score: float = 0.0
    
    # Detailed metrics
    auc_mean: float = 0.0
    auc_std: float = 0.0
    pr_auc_mean: float = 0.0
    pr_auc_std: float = 0.0
    psi_score: float = 0.0
    flip_rate: float = 0.0
    prevalence: float = 0.0
    mutual_information: float = 0.0
    information_coefficient: float = 0.0
    
    # Model performance
    baseline_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Metadata
    n_samples: int = 0
    n_features: int = 0
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class LabelQualityScorer(BaseStep):
    """
    Label Quality Scorer for Volatility-Aware Labeling
    
    This class implements comprehensive label quality assessment to ensure labels are
    learnable by ML models and generalize well.
    Inherits from BaseStep for standardized pipeline integration.
    
    Key Features:
    1. **Predictability Assessment**: Uses baseline models to measure learnability
    2. **Stability Measurement**: Evaluates consistency across rolling folds
    3. **Consistency Evaluation**: Measures mutual information between labels
    4. **Balance Assessment**: Ensures proper class distribution
    5. **SNR Proxy Calculation**: Uses information coefficient for signal quality
    6. **Composite Scoring**: Combines all metrics into Label Quality Score (LQS)
    """
    
    def __init__(self, config: Optional[QualityScoringConfig] = None):
        """Initialize label quality scorer."""
        super().__init__()
        self.config = config or QualityScoringConfig()
        self.logger = logging.getLogger('LabelQualityScorer')

        # Initialize matrix operations for vectorized computations
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = UnifiedMatrixOperations()
            self.tprint_info("   → Matrix operations: Available")
        else:
            self.matrix_ops = None
            self.tprint_warning("   → Matrix operations: Not available, using fallback")

        # Initialize Pareto optimizer if available
        if PARETO_OPTIMIZER_AVAILABLE and self.config.enable_pareto_optimization:
            self.pareto_optimizer = ParetoFront()
            self.tprint_info("   → Pareto optimizer: Available for multi-objective optimization")
        else:
            self.pareto_optimizer = None

        # Initialize CV utilities
        if CV_UTILITIES_AVAILABLE:
            self.cv_validator = CrossValidator()
            self.tprint_info("   → CV utilities: Available")
        else:
            self.cv_validator = None

        # Initialize OOF stacking
        if OOF_AVAILABLE:
            self.oof_manager = OOFStackingEnsembleManager()
            self.tprint_info("   → OOF stacking: Available")
        else:
            self.oof_manager = None

        self.tprint_info("📊 Label Quality Scorer initialized")
        self.tprint_info(f"   → Baseline models: {self.config.baseline_models}")
        self.tprint_info(f"   → CV splits: {self.config.n_splits}")
        self.tprint_info(f"   → Optimization: {self.config.enable_optimization}")
        self.tprint_info(f"   → Pareto optimization: {self.config.enable_pareto_optimization}")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the label quality scoring step.
        
        Args:
            config: Configuration dictionary containing:
                - labels: DataFrame with label data
                - confidence_scores: DataFrame with confidence scores
                - eligibility_masks: DataFrame with eligibility masks
                - bars: DataFrame with bar data for feature engineering
                - symbol: Optional symbol for context
                - exchange: Optional exchange for context
                - information: Optional information for context
                - direction: Optional direction for context
                - model: Optional model type for context
        
        Returns:
            Dictionary containing:
                - success: Boolean indicating success
                - quality_results: Dictionary of quality metrics per target
                - overall_quality: Overall quality assessment
                - artifacts: List of generated artifacts
        """
        try:
            # Set context for enhanced file naming and operations
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information=config.get('information'),
                direction=config.get('direction', 'long'),
                model=config.get('model', 'Analyst')
            )
            
            # Extract parameters from config
            labels = config.get('labels')
            confidence_scores = config.get('confidence_scores', pd.DataFrame())
            eligibility_masks = config.get('eligibility_masks', pd.DataFrame())
            bars = config.get('bars', pd.DataFrame())
            
            if labels is None:
                return {
                    'success': False,
                    'error': 'Missing required parameter: labels'
                }
            
            # Validate inputs
            if not isinstance(labels, pd.DataFrame):
                return {
                    'success': False,
                    'error': 'labels must be a pandas DataFrame'
                }
            
            # Assess quality
            quality_results = self.assess_quality(
                labels=labels,
                confidence_scores=confidence_scores,
                eligibility_masks=eligibility_masks,
                bars=bars
            )
            
            # Calculate overall quality
            overall_quality = self._calculate_overall_quality(quality_results)
            
            # Save artifacts
            artifacts = []
            
            # Prepare quality results data
            quality_data = {target: {
                'n_samples': metrics.n_samples,
                'n_features': metrics.n_features,
                'predictability': metrics.predictability,
                'stability': metrics.stability,
                'consistency': metrics.consistency,
                'balance': metrics.balance,
                'snr_proxy': metrics.snr_proxy,
                'sparsity': metrics.sparsity,
                'lqs_score': metrics.lqs_score,
                'class_prevalence': metrics.prevalence
            } for target, metrics in quality_results.items()}
            
            # Preview quality results
            self.tprint_data_format(quality_data, "label_quality_results")
            
            # Save quality results
            quality_path = self._save_metadata(
                quality_data,
                'label_quality_results'
            )
            if quality_path:
                artifacts.append(quality_path)
            
            # Preview overall quality
            self.tprint_data_format(overall_quality, "overall_quality_assessment")
            
            # Save overall quality
            overall_path = self._save_metadata(
                overall_quality,
                'overall_quality_assessment'
            )
            if overall_path:
                artifacts.append(overall_path)
            
            # Generate outcome file
            outcome_content = self._generate_outcome_content(quality_results, overall_quality, artifacts)
            self._save_outcome_file(outcome_content, 'label_quality_scoring_outcome')
            
            return {
                'success': True,
                'quality_results': quality_results,
                'overall_quality': overall_quality,
                'artifacts': artifacts
            }
            
        except Exception as e:
            error_msg = f"Label quality scoring failed: {str(e)}"
            self.tprint_error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def _calculate_overall_quality(self, quality_results: Dict[str, QualityMetrics]) -> Dict[str, Any]:
        """Calculate overall quality metrics across all targets."""
        if not quality_results:
            return {'overall_score': 0.0, 'n_targets': 0}
        
        # Calculate weighted average scores
        total_samples = sum(metrics.n_samples for metrics in quality_results.values())
        if total_samples == 0:
            return {'overall_score': 0.0, 'n_targets': len(quality_results)}
        
        weighted_scores = {}
        for metric in ['predictability', 'stability', 'consistency', 
                      'balance', 'snr_proxy', 'sparsity', 'lqs_score']:
            weighted_sum = sum(
                getattr(metrics, metric, 0) * metrics.n_samples 
                for metrics in quality_results.values()
            )
            weighted_scores[metric] = weighted_sum / total_samples
        
        return {
            'overall_score': weighted_scores.get('lqs_score', 0),
            'n_targets': len(quality_results),
            'total_samples': total_samples,
            **weighted_scores
        }
    
    def _generate_outcome_content(self, quality_results: Dict[str, QualityMetrics], 
                                overall_quality: Dict[str, Any], artifacts: List[str]) -> str:
        """Generate outcome file content."""
        content = f"""# Label Quality Scoring Outcome

## Summary
- **Status**: Success
- **Targets Assessed**: {len(quality_results)}
- **Total Samples**: {overall_quality.get('total_samples', 0)}
- **Overall Quality Score**: {overall_quality.get('overall_score', 0):.3f}
- **Artifacts Generated**: {len(artifacts)}

## Quality Metrics by Target
"""
        
        for target, metrics in quality_results.items():
            content += f"""
### {target}
- **Samples**: {metrics.n_samples}
- **Features**: {metrics.n_features}
- **Predictability**: {metrics.predictability:.3f}
- **Stability**: {metrics.stability:.3f}
- **Consistency**: {metrics.consistency:.3f}
- **Balance**: {metrics.balance:.3f}
- **SNR Proxy**: {metrics.snr_proxy:.3f}
- **Sparsity**: {metrics.sparsity:.3f}
- **LQS Score**: {metrics.lqs_score:.3f}
- **Prevalence**: {metrics.prevalence:.3f}
"""
        
        content += f"""
## Overall Quality Assessment
- **Overall Score**: {overall_quality.get('overall_score', 0):.3f}
- **Predictability**: {overall_quality.get('predictability', 0):.3f}
- **Stability**: {overall_quality.get('stability', 0):.3f}
- **Consistency**: {overall_quality.get('consistency', 0):.3f}
- **Balance**: {overall_quality.get('balance', 0):.3f}
- **SNR Proxy**: {overall_quality.get('snr_proxy', 0):.3f}
- **Sparsity**: {overall_quality.get('sparsity', 0):.3f}

## Generated Artifacts
{chr(10).join(f"- {artifact}" for artifact in artifacts)}
"""
        
        return content
    
    def assess_quality(self, labels: pd.DataFrame, confidence_scores: pd.DataFrame,
                      eligibility_masks: pd.DataFrame, bars: pd.DataFrame) -> Dict[str, QualityMetrics]:
        """
        Assess quality of labels across all targets.
        
        Args:
            labels: Label DataFrame with target columns
            confidence_scores: Confidence scores DataFrame
            eligibility_masks: Eligibility masks DataFrame
            bars: Cleaned bars for feature engineering
            
        Returns:
            Dictionary mapping target names to QualityMetrics
        """
        start_time = datetime.now()
        self.tprint_info("📊 Assessing label quality")
        
        quality_results = {}
        
        try:
            # Get target columns
            target_columns = [col for col in labels.columns if 'target' in col.lower()]
            
            if not target_columns:
                self.tprint_warning("⚠️ No target columns found")
                return quality_results
            
            # Process each target
            for target_col in target_columns:
                self.tprint_info(f"📈 Assessing quality for target: {target_col}")
                
                # Extract target data
                target_labels = labels[target_col].dropna()
                target_confidence = confidence_scores.get(target_col, pd.Series(dtype=float))
                target_eligibility = eligibility_masks.get(target_col, pd.Series(True, index=target_labels.index))
                
                # Filter by eligibility
                eligible_mask = target_eligibility & target_eligibility.notna()
                if not eligible_mask.any():
                    self.tprint_warning(f"⚠️ No eligible samples for target {target_col}")
                    continue
                
                target_labels_eligible = target_labels[eligible_mask]
                
                # Check minimum samples
                if len(target_labels_eligible) < self.config.min_samples_for_evaluation:
                    self.tprint_warning(f"⚠️ Insufficient samples for target {target_col}: {len(target_labels_eligible)}")
                    continue
                
                # Assess quality for this target
                quality_metrics = self._assess_single_target_quality(
                    target_labels_eligible, target_confidence, bars, target_col
                )
                
                quality_results[target_col] = quality_metrics
            
        except Exception as e:
            self.tprint_error(f"❌ Quality assessment failed: {e}")
            return quality_results
        
        # Second pass: build LQS via EWM
        if quality_results:
            components = self.config.lqs_components
            M, _ = self._normalize_metrics_matrix(quality_results, components)
            w = self._entropy_weights(M)
            lqs = (M * w).sum(axis=1)
            
            # write back per target
            for t, score in lqs.items():
                quality_results[t].lqs_score = float(score)
            self.tprint_info(f"   → LQS weights (EWM): {w.to_dict()}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.tprint_success("✅ Quality assessment completed")
        self.tprint_info(f"   → Processing time: {processing_time:.2f}s")
        self.tprint_info(f"   → Targets assessed: {len(quality_results)}")
        
        return quality_results
    
    def _assess_single_target_quality(self, target_labels: pd.Series, confidence_scores: pd.Series,
                                    bars: pd.DataFrame, target_name: str) -> QualityMetrics:
        """Assess quality for a single target."""
        try:
            start_time = datetime.now()
            
            # Initialize metrics
            metrics = QualityMetrics(
                n_samples=len(target_labels),
                n_features=0
            )
            
            # Generate features if enabled
            if self.config.enable_feature_engineering:
                features = self._generate_baseline_features(bars, target_labels.index)
                metrics.n_features = features.shape[1] if not features.empty else 0
            else:
                features = pd.DataFrame()
            
            # 1. Predictability assessment
            tprint_info(f"   📈 Assessing predictability for {target_name}")
            predictability_metrics = self._assess_predictability(target_labels, features)
            metrics.predictability = predictability_metrics['predictability']
            metrics.auc_mean = predictability_metrics['auc_mean']
            metrics.auc_std = predictability_metrics['auc_std']
            metrics.pr_auc_mean = predictability_metrics['pr_auc_mean']
            metrics.pr_auc_std = predictability_metrics['pr_auc_std']
            metrics.prevalence = float(predictability_metrics.get('prevalence', float((target_labels > 0).mean())))
            metrics.baseline_performance = predictability_metrics['baseline_performance']
            
            # 2. Stability assessment
            tprint_info(f"   📊 Assessing stability for {target_name}")
            stability_metrics = self._assess_stability(target_labels, features)
            metrics.stability = stability_metrics['stability']
            metrics.psi_score = stability_metrics['psi_score']
            
            # 3. Consistency assessment
            tprint_info(f"   🔗 Assessing consistency for {target_name}")
            consistency_metrics = self._assess_consistency(target_labels)
            metrics.consistency = consistency_metrics['consistency']
            metrics.mutual_information = consistency_metrics['mutual_information']
            metrics.flip_rate = consistency_metrics['flip_rate']
            
            # 4. Balance assessment
            tprint_info(f"   ⚖️ Assessing balance for {target_name}")
            balance_metrics = self._assess_balance(target_labels)
            metrics.balance = balance_metrics['balance']
            metrics.prevalence = float(balance_metrics.get('prevalence', metrics.prevalence))
            
            # 5. SNR proxy assessment
            tprint_info(f"   📡 Assessing SNR proxy for {target_name}")
            snr_metrics = self._assess_snr_proxy(target_labels, features)
            metrics.snr_proxy = snr_metrics['snr_proxy']
            metrics.information_coefficient = snr_metrics['information_coefficient']
            
            # 6. Sparsity assessment
            tprint_info(f"   📊 Assessing sparsity for {target_name}")
            sparsity_metrics = self._assess_sparsity(target_labels)
            metrics.sparsity = sparsity_metrics['sparsity']
            
            # 7. Refine stability using fold AUC std if present
            if metrics.auc_std > 0:
                metrics.stability = float(1.0 / (1.0 + metrics.auc_std + metrics.psi_score))
            
            # 8. Calculate composite LQS score (will be overridden by EWM)
            metrics.lqs_score = self._calculate_lqs_score(metrics)
            
            # Calculate processing time
            metrics.processing_time = (datetime.now() - start_time).total_seconds()
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Error assessing quality for {target_name}: {e}")
            return QualityMetrics(n_samples=len(target_labels))
    
    def _generate_baseline_features(self, bars: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
        """Generate baseline features for quality assessment."""
        try:
            if bars.empty or len(target_index) == 0:
                return pd.DataFrame()
            
            # Align bars with target index
            bars_aligned = bars.reindex(target_index, method='ffill')
            
            if bars_aligned.empty:
                return pd.DataFrame()
            
            # Vectorized feature calculation using matrix operations where possible
            close_prices = bars_aligned['close'].values
            volume_values = bars_aligned['volume'].values
            high_prices = bars_aligned['high'].values
            low_prices = bars_aligned['low'].values
            open_prices = bars_aligned['open'].values

            features_data = {}

            # Price-based features (vectorized) - use trailing operations
            close = pd.Series(close_prices, index=target_index)
            ret = close.pct_change().fillna(0.0)
            features_data['returns'] = ret.values

            # Log returns
            log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            features_data['log_returns'] = log_ret.values

            # Volatility (rolling std) - use vectorized operations
            returns_series = pd.Series(features_data['returns'], index=target_index)
            features_data['volatility'] = returns_series.rolling(self.config.feature_window).std().fillna(0).values

            # Price momentum
            shifted_close = close.shift(self.config.feature_window)
            features_data['price_momentum'] = (close / shifted_close - 1.0).fillna(0.0).values

            # Volume-based features (vectorized)
            rolling_volume_mean = pd.Series(volume_values, index=target_index).rolling(self.config.feature_window).mean().fillna(method='bfill').values
            features_data['volume_ratio'] = np.divide(volume_values, rolling_volume_mean, out=np.zeros_like(volume_values), where=rolling_volume_mean!=0)

            # Volume momentum
            features_data['volume_momentum'] = np.diff(volume_values, prepend=volume_values[0]) / volume_values[:-1]
            features_data['volume_momentum'] = np.concatenate([[0], features_data['volume_momentum']])

            # OHLC-based features (vectorized)
            features_data['high_low_ratio'] = (high_prices - low_prices) / close_prices
            features_data['close_open_ratio'] = close_prices / open_prices - 1

            # Technical indicators
            rolling_close_mean = pd.Series(close_prices, index=target_index).rolling(self.config.feature_window).mean().fillna(method='bfill').values
            features_data['sma_ratio'] = close_prices / rolling_close_mean

            # RSI calculation
            features_data['rsi'] = self._rsi_wilder(close, self.config.feature_window).fillna(50.0).values

            # Create DataFrame efficiently
            features = pd.DataFrame(features_data, index=target_index)
            
            # Select top features if too many
            if features.shape[1] > self.config.n_features:
                # Use correlation with target to select features
                # For now, just take the first n_features
                features = features.iloc[:, :self.config.n_features]
            
            return features
            
        except Exception as e:
            tprint_warning(f"⚠️ Error generating baseline features: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series(0, index=prices.index)

    def _calculate_rsi_vectorized(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate RSI indicator using vectorized operations."""
        try:
            # Vectorized RSI calculation for better performance
            deltas = np.diff(prices)
            deltas = np.concatenate([[0], deltas])  # Pad first difference

            # Calculate gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            # Rolling means using vectorized operations
            # Use cumulative approach for efficiency
            gains_rolling = np.zeros_like(gains, dtype=float)
            losses_rolling = np.zeros_like(losses, dtype=float)

            # Vectorized rolling mean calculation
            for i in range(len(gains)):
                if i < window:
                    gains_rolling[i] = np.mean(gains[:i+1]) if i > 0 else 0
                    losses_rolling[i] = np.mean(losses[:i+1]) if i > 0 else 0
                else:
                    gains_rolling[i] = np.mean(gains[i-window+1:i+1])
                    losses_rolling[i] = np.mean(losses[i-window+1:i+1])

            # Calculate RS and RSI
            rs = np.divide(gains_rolling, losses_rolling, out=np.ones_like(gains_rolling), where=losses_rolling!=0)
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception as e:
            tprint_warning(f"⚠️ Vectorized RSI calculation failed: {e}")
            # Fallback to pandas implementation
            prices_series = pd.Series(prices)
            return self._calculate_rsi(prices_series, window).values
    
    def _rsi_wilder(self, close: pd.Series, window: int) -> pd.Series:
        """Calculate RSI using Wilder's EWMA method (no leakage)."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - 100 / (1 + rs)
        return rsi.fillna(50.0)
    
    def _assess_predictability(self, target_labels: pd.Series, features: pd.DataFrame) -> Dict[str, Any]:
        """Assess predictability using baseline models."""
        try:
            if features.empty or len(target_labels) < 50:
                return {
                    'predictability': 0.0,
                    'auc_mean': 0.0,
                    'auc_std': 0.0,
                    'pr_auc_mean': 0.0,
                    'pr_auc_std': 0.0,
                    'baseline_performance': {}
                }
            
            # Align features with labels
            common_index = target_labels.index.intersection(features.index)
            if len(common_index) < 50:
                return {
                    'predictability': 0.0,
                    'auc_mean': 0.0,
                    'auc_std': 0.0,
                    'pr_auc_mean': 0.0,
                    'pr_auc_std': 0.0,
                    'baseline_performance': {}
                }
            
            y = target_labels.loc[common_index]
            X = features.loc[common_index]
            
            # Remove any remaining NaN values
            valid_mask = y.notna() & X.notna().all(axis=1)
            y = y[valid_mask]
            X = X[valid_mask]
            
            if len(y) < 50:
                return {
                    'predictability': 0.0,
                    'auc_mean': 0.0,
                    'auc_std': 0.0,
                    'pr_auc_mean': 0.0,
                    'pr_auc_std': 0.0,
                    'baseline_performance': {}
                }
            
            # Convert to binary classification if needed
            if y.nunique() > 2:
                # Convert to binary: positive vs non-positive
                y_binary = (y > 0).astype(int)
            else:
                y_binary = y.astype(int)
            
            # Check if we have both classes
            if y_binary.nunique() < 2:
                return {
                    'predictability': 0.0,
                    'auc_mean': 0.0,
                    'auc_std': 0.0,
                    'pr_auc_mean': 0.0,
                    'pr_auc_std': 0.0,
                    'baseline_performance': {}
                }
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            auc_scores = []
            pr_auc_scores = []
            baseline_performance = {}
            
            for model_name in self.config.baseline_models:
                model_auc_scores = []
                model_pr_auc_scores = []
                
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y_binary.iloc[train_idx], y_binary.iloc[test_idx]
                    
                    # Skip fold if single class
                    if y_train.nunique() < 2 or y_test.nunique() < 2:
                        continue
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train model
                    if model_name == 'logistic':
                        model = LogisticRegression(random_state=self.config.random_state, max_iter=1000)
                    elif model_name == 'random_forest':
                        model = RandomForestClassifier(n_estimators=50, random_state=self.config.random_state)
                    else:
                        continue
                    
                    try:
                        model.fit(X_train_scaled, y_train)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                        
                        # Calculate AUC
                        if len(np.unique(y_test)) > 1:
                            auc_score = roc_auc_score(y_test, y_pred_proba)
                            model_auc_scores.append(auc_score)
                            
                            # Calculate PR-AUC
                            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                            pr_auc_score = auc(recall, precision)
                            model_pr_auc_scores.append(pr_auc_score)
                        
                    except Exception as e:
                        tprint_warning(f"⚠️ Error training {model_name}: {e}")
                        continue
                
                if model_auc_scores:
                    baseline_performance[model_name] = {
                        'auc_mean': np.mean(model_auc_scores),
                        'auc_std': np.std(model_auc_scores),
                        'pr_auc_mean': np.mean(model_pr_auc_scores),
                        'pr_auc_std': np.std(model_pr_auc_scores)
                    }
                    auc_scores.extend(model_auc_scores)
                    pr_auc_scores.extend(model_pr_auc_scores)
            
            # Calculate overall metrics
            auc_mean = float(np.mean(auc_scores)) if auc_scores else 0.0
            auc_std = float(np.std(auc_scores)) if auc_scores else 0.0
            pr_auc_mean = float(np.mean(pr_auc_scores)) if pr_auc_scores else 0.0
            pr_auc_std = float(np.std(pr_auc_scores)) if pr_auc_scores else 0.0
            prevalence = float(y_binary.mean()) if len(y_binary) else 0.0
            
            # Adjust for trivial baselines
            auc_adj = max(0.0, 2.0 * (auc_mean - 0.5))
            denom = max(1e-12, 1.0 - prevalence)
            prauc_adj = max(0.0, (pr_auc_mean - prevalence) / denom)
            
            # Harmonic mean (if both > 0)
            if auc_adj > 0 and prauc_adj > 0:
                predictability = 2 * (auc_adj * prauc_adj) / (auc_adj + prauc_adj)
            else:
                predictability = max(auc_adj, prauc_adj)
            
            return {
                'predictability': predictability,
                'auc_mean': auc_mean,
                'auc_std': auc_std,
                'pr_auc_mean': pr_auc_mean,
                'pr_auc_std': pr_auc_std,
                'prevalence': prevalence,
                'baseline_performance': baseline_performance
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in predictability assessment: {e}")
            return {
                'predictability': 0.0,
                'auc_mean': 0.0,
                'auc_std': 0.0,
                'pr_auc_mean': 0.0,
                'pr_auc_std': 0.0,
                'baseline_performance': {}
            }
    
    def _assess_stability(self, target_labels: pd.Series, features: pd.DataFrame) -> Dict[str, float]:
        """Assess stability using PSI and rolling window analysis."""
        try:
            if len(target_labels) < 100:
                return {'stability': 0.0, 'psi_score': 0.0}
            
            # Calculate PSI (Population Stability Index)
            psi_score = self._calculate_psi(target_labels)
            
            # Calculate rolling stability
            window_size = min(50, len(target_labels) // 4)
            if window_size < 10:
                return {'stability': 1.0 - psi_score, 'psi_score': psi_score}
            
            # Rolling window analysis
            rolling_means = target_labels.rolling(window=window_size).mean()
            rolling_stds = target_labels.rolling(window=window_size).std()
            
            # Stability based on raw signals without heuristics
            stability = 1.0 / (1.0 + (rolling_means.std() / (abs(rolling_means.mean()) + 1e-12)) + (rolling_stds.std() / (rolling_stds.mean() + 1e-12)))
            
            return {
                'stability': float(np.clip(stability, 0.0, 1.0)),
                'psi_score': float(psi_score)
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in stability assessment: {e}")
            return {'stability': 0.0, 'psi_score': 0.0}
    
    def _calculate_psi(self, series: pd.Series) -> float:
        """Calculate Population Stability Index."""
        try:
            if len(series) < 20:
                return 0.0
            
            # Split into two halves
            mid_point = len(series) // 2
            expected = series.iloc[:mid_point]
            actual = series.iloc[mid_point:]
            
            # Create bins - handle binary series
            if series.nunique() <= 2:
                bins = np.array([-np.inf, 0.5, np.inf])
            else:
                bins = np.linspace(series.min(), series.max(), 11)
            
            # Calculate distributions
            expected_dist = np.histogram(expected, bins=bins)[0] / len(expected)
            actual_dist = np.histogram(actual, bins=bins)[0] / len(actual)
            
            # Avoid division by zero
            expected_dist = np.where(expected_dist == 0, 1e-6, expected_dist)
            actual_dist = np.where(actual_dist == 0, 1e-6, actual_dist)
            
            # Calculate PSI
            psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
            
            return max(0.0, psi)
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating PSI: {e}")
            return 0.0
    
    def _assess_consistency(self, target_labels: pd.Series) -> Dict[str, float]:
        """Assess consistency using mutual information and flip rate."""
        try:
            if len(target_labels) < 20:
                return {'consistency': 0.0, 'mutual_information': 0.0, 'flip_rate': 0.0}
            
            # Calculate flip rate (label changes)
            label_changes = (target_labels != target_labels.shift(1)).sum()
            flip_rate = label_changes / (len(target_labels) - 1)
            
            # Calculate mutual information with lagged labels
            if len(target_labels) > 1:
                lagged_labels = target_labels.shift(1).dropna()
                current_labels = target_labels.iloc[1:]
                
                # Calculate mutual information
                mutual_info = self._calculate_mutual_information(current_labels, lagged_labels)
            else:
                mutual_info = 0.0
            
            # Normalize MI to [0,1] via theoretical max ≈ log2(bins)
            bins = 10
            mi_norm = min(1.0, mutual_info / max(1e-12, np.log2(bins)))
            # Combine without hand weights (geometric mean)
            consistency = float(np.sqrt(max(0.0, mi_norm) * max(0.0, 1.0 - flip_rate)))
            
            return {
                'consistency': consistency,
                'mutual_information': mutual_info,
                'flip_rate': flip_rate
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in consistency assessment: {e}")
            return {'consistency': 0.0, 'mutual_information': 0.0, 'flip_rate': 0.0}
    
    def _calculate_mutual_information(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate mutual information between two series."""
        try:
            # Align series
            common_index = x.index.intersection(y.index)
            if len(common_index) < 10:
                return 0.0
            
            x_aligned = x.loc[common_index]
            y_aligned = y.loc[common_index]
            
            # Create joint distribution
            x_bins = pd.cut(x_aligned, bins=10, labels=False, duplicates='drop')
            y_bins = pd.cut(y_aligned, bins=10, labels=False, duplicates='drop')
            
            # Calculate joint and marginal distributions
            joint_dist = pd.crosstab(x_bins, y_bins, normalize=True)
            x_dist = x_bins.value_counts(normalize=True)
            y_dist = y_bins.value_counts(normalize=True)
            
            # Calculate mutual information
            mi = 0.0
            for i in joint_dist.index:
                for j in joint_dist.columns:
                    p_xy = joint_dist.loc[i, j]
                    p_x = x_dist.get(i, 0)
                    p_y = y_dist.get(j, 0)
                    
                    if p_xy > 0 and p_x > 0 and p_y > 0:
                        mi += p_xy * np.log2(p_xy / (p_x * p_y))
            
            return max(0.0, mi)
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating mutual information: {e}")
            return 0.0
    
    def _assess_balance(self, target_labels: pd.Series) -> Dict[str, float]:
        """Assess class balance."""
        try:
            if len(target_labels) == 0:
                return {'balance': 0.0, 'class_balance': 0.0}
            
            # Calculate class distribution
            value_counts = target_labels.value_counts()
            total_samples = len(target_labels)
            
            # Calculate balance score
            if len(value_counts) == 1:
                # Only one class
                balance = 0.0
                class_balance = 1.0 if value_counts.iloc[0] == total_samples else 0.0
            else:
                # Multiple classes
                max_class_ratio = value_counts.max() / total_samples
                min_class_ratio = value_counts.min() / total_samples
                
                # Balance score: closer to 0.5 is better
                class_balance = max_class_ratio
                balance = 1.0 - abs(max_class_ratio - 0.5) * 2
            
            return {
                'balance': max(0.0, min(1.0, balance)),
                'class_balance': class_balance,
                'prevalence': class_balance
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in balance assessment: {e}")
            return {'balance': 0.0, 'class_balance': 0.0}
    
    def _assess_snr_proxy(self, target_labels: pd.Series, features: pd.DataFrame) -> Dict[str, float]:
        """Assess signal-to-noise ratio proxy using information coefficient."""
        try:
            if features.empty or len(target_labels) < 20:
                return {'snr_proxy': 0.0, 'information_coefficient': 0.0}
            
            # Align features with labels
            common_index = target_labels.index.intersection(features.index)
            if len(common_index) < 20:
                return {'snr_proxy': 0.0, 'information_coefficient': 0.0}
            
            y = target_labels.loc[common_index]
            X = features.loc[common_index]
            
            # Align indices and guard against NaNs
            mask = y.notna()
            y = y[mask]
            X = X.loc[mask]
            
            # Calculate information coefficient for each feature
            ic_scores = []
            for col in X.columns:
                if X[col].notna().sum() > 10:
                    try:
                        # Calculate Spearman correlation
                        corr, _ = spearmanr(X[col].dropna(), y.loc[X[col].dropna().index])
                        if not np.isnan(corr):
                            ic_scores.append(abs(corr))
                    except Exception:
                        continue
            
            # Calculate average IC
            avg_ic = np.mean(ic_scores) if ic_scores else 0.0
            
            # SNR proxy is the average IC
            snr_proxy = avg_ic
            
            return {
                'snr_proxy': snr_proxy,
                'information_coefficient': avg_ic
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in SNR proxy assessment: {e}")
            return {'snr_proxy': 0.0, 'information_coefficient': 0.0}
    
    def _assess_sparsity(self, target_labels: pd.Series) -> Dict[str, float]:
        """Assess label sparsity (minimum positive class ratio)."""
        try:
            if len(target_labels) == 0:
                return {'sparsity': 0.0}
            
            # Calculate positive class ratio
            positive_ratio = (target_labels > 0).sum() / len(target_labels)
            
            # Sparsity score: monotone function without hard thresholds
            target_floor = 0.05  # 5% reference
            sparsity_score = float(np.clip(positive_ratio / target_floor, 0.0, 1.0))
            
            return {
                'sparsity': sparsity_score
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in sparsity assessment: {e}")
            return {'sparsity': 0.0}
    
    def _calculate_lqs_score(self, metrics: QualityMetrics) -> float:
        """Calculate composite Label Quality Score (LQS)."""
        try:
            # Get weights
            weights = self.config.lqs_weights
            
            # Calculate weighted score
            lqs_score = (
                weights['predictability'] * metrics.predictability +
                weights['stability'] * metrics.stability +
                weights['consistency'] * metrics.consistency +
                weights['balance'] * metrics.balance +
                weights['snr_proxy'] * metrics.snr_proxy +
                weights['sparsity'] * metrics.sparsity
            )
            
            return max(0.0, min(1.0, lqs_score))
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating LQS score: {e}")
            return 0.0
    
    def _normalize_metrics_matrix(self, rows: Dict[str, QualityMetrics], keys: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Build and normalize metrics matrix for EWM calculation."""
        # Build matrix T x K
        data = []
        idx = []
        for t, m in rows.items():
            row = [max(0.0, float(getattr(m, k, 0.0))) for k in keys]
            data.append(row)
            idx.append(t)
        M = pd.DataFrame(data, index=idx, columns=keys)
        
        # Min-max per column with epsilon
        eps = 1e-12
        for k in keys:
            col = M[k].astype(float)
            lo, hi = col.min(), col.max()
            if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
                M[k] = 0.0  # no dispersion -> no contribution
            else:
                M[k] = (col - lo) / (hi - lo)
        
        return M, M.sum(axis=0).replace(0, eps)
    
    def _entropy_weights(self, M: pd.DataFrame) -> pd.Series:
        """Calculate entropy weights for EWM."""
        # EWM: w_j ∝ 1 - entropy_j
        eps = 1e-12
        # normalize each column to probabilities per metric
        P = M.div(M.sum(axis=0) + eps, axis=1).replace(0, eps)
        k = M.shape[0]
        H = - (P * np.log(P)).sum(axis=0) / np.log(k + eps)
        d = 1.0 - H
        w = d / (d.sum() + eps)
        return w
    
    def optimize_target_selection_pareto(self, quality_results: Dict[str, QualityMetrics]) -> List[str]:
        """Use Pareto optimization to select optimal targets."""
        try:
            if not self.pareto_optimizer or not quality_results:
                # Fallback to LQS-based selection
                return self._select_targets_by_lqs(quality_results)
            
            # Convert quality results to Pareto solutions
            solutions = []
            for target_name, metrics in quality_results.items():
                solution_metrics = {
                    'predictability': metrics.predictability,
                    'stability': metrics.stability,
                    'consistency': metrics.consistency,
                    'balance': metrics.balance,
                    'sparsity': metrics.sparsity
                }
                
                solution = Solution(
                    metrics=solution_metrics,
                    params={'target_name': target_name}
                )
                solutions.append(solution)
            
            # Define objectives (all maximization)
            objectives = {
                'predictability': 'max',
                'stability': 'max',
                'consistency': 'max',
                'balance': 'max',
                'sparsity': 'max'
            }
            
            # Compute Pareto front
            pareto_solutions = self.pareto_optimizer.optimize(solutions, objectives)
            
            # Extract target names from Pareto solutions
            selected_targets = []
            for solution in pareto_solutions:
                if solution.params and 'target_name' in solution.params:
                    selected_targets.append(solution.params['target_name'])
            
            tprint_info(f"📊 Pareto optimization selected {len(selected_targets)} targets")
            return selected_targets
            
        except Exception as e:
            tprint_warning(f"⚠️ Pareto optimization failed: {e}")
            return self._select_targets_by_lqs(quality_results)
    
    def _select_targets_by_lqs(self, quality_results: Dict[str, QualityMetrics]) -> List[str]:
        """Fallback method: select targets by LQS score."""
        try:
            # Sort by LQS score
            sorted_targets = sorted(
                quality_results.items(),
                key=lambda x: x[1].lqs_score,
                reverse=True
            )
            
            # Select top targets
            selected_targets = [name for name, _ in sorted_targets[:5]]  # Top 5
            return selected_targets
            
        except Exception as e:
            tprint_warning(f"⚠️ LQS-based selection failed: {e}")
            return list(quality_results.keys())


# Convenience functions
def create_label_quality_scorer(config: Optional[QualityScoringConfig] = None) -> LabelQualityScorer:
    """Create label quality scorer with specified configuration."""
    return LabelQualityScorer(config)


def assess_label_quality(labels: pd.DataFrame, confidence_scores: pd.DataFrame,
                        eligibility_masks: pd.DataFrame, bars: pd.DataFrame,
                        config: Optional[QualityScoringConfig] = None) -> Dict[str, QualityMetrics]:
    """Assess label quality with default configuration."""
    scorer = LabelQualityScorer(config)
    return scorer.assess_quality(labels, confidence_scores, eligibility_masks, bars)