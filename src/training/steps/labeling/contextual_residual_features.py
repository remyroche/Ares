"""
Contextual Residual Features for Meta-Learning

This module implements a 3-step system to generate contextual residual features
from base model predictions, enabling the meta-learner to understand the "State of the Experts."

Step 1: Outcome Harmonization
- Convert all base model outputs to consistent space (direction [-1,1] or probability [0,1])
- Align predictions against meta-label from Triple Barrier

Step 2: Generate Contextual Residual Features  
- Residual (R): y_true - y_pred (model accuracy)
- Bias (EMA): Winning/losing streaks with exponential decay
- Volatility (Rolling Std): Performance stability
- Reliability (CUSUM): Cumulative error sum for structural break detection

Step 3: Pruning Engine Integration
- Use De Prado engine on 60 residual + market features
- ONC clustering for redundancy reduction
- Advanced MDI for predictive power
- Root proximity for hierarchy analysis

Usage:
- Enhances meta-learning by tracking model expertise
- Enables adaptive weighting based on model performance
- Detects structural breaks in model logic
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy.stats import entropy
import warnings
from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error

EPS = 1e-12

class ContextualResidualFeatureGenerator:
    """
    Generates contextual residual features from base model predictions.
    
    This class implements the 3-step process to create 60 meta-features
    from 20 base model predictions, describing the "State of the Experts."
    """
    
    def __init__(
        self,
        harmonization_type: str = "direction",  # "direction" or "probability"
        bias_window: int = 20,
        volatility_window: int = 30,
        reliability_window: int = 50,
        cusum_threshold: float = 2.0,
        min_samples: int = 100
    ):
        """
        Initialize contextual residual feature generator.
        
        Args:
            harmonization_type: Type of harmonization ("direction" or "probability")
            bias_window: Window size for EMA bias calculation
            volatility_window: Window size for rolling volatility
            reliability_window: Window size for CUSUM reliability
            cusum_threshold: Threshold for CUSUM signal generation
            min_samples: Minimum samples required for reliable calculation
        """
        self.harmonization_type = harmonization_type
        self.bias_window = bias_window
        self.volatility_window = volatility_window
        self.reliability_window = reliability_window
        self.cusum_threshold = cusum_threshold
        self.min_samples = min_samples
        
        # Store computed features for analysis
        self.harmonized_predictions_ = None
        self.residual_features_ = None
        self.feature_stats_ = None
        
    def harmonize_base_model_predictions(
        self,
        predictions_df: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """
        Step 1: Harmonize all base model predictions to consistent space.
        
        Args:
            predictions_df: DataFrame with base model predictions
            target_col: Target column name (meta-label from Triple Barrier)
            
        Returns:
            DataFrame with harmonized predictions
        """
        tprint_info("🔧 Step 1: Harmonizing Base Model Predictions...")
        
        if len(predictions_df) < self.min_samples:
            tprint_error(f"❌ Insufficient samples: {len(predictions_df)} < {self.min_samples}")
            raise ValueError(f"Insufficient samples: {len(predictions_df)} < {self.min_samples}")
        
        if target_col not in predictions_df.columns:
            tprint_error(f"❌ Target column {target_col} not found in predictions")
            raise ValueError(f"Target column {target_col} not found")
        
        # Extract base model columns (exclude target)
        base_cols = [col for col in predictions_df.columns if col != target_col]
        harmonized = predictions_df[[target_col]].copy()
        
        tprint_info(f"📊 Harmonizing {len(base_cols)} base model predictions")
        
        for col in base_cols:
            if col not in predictions_df.columns:
                tprint_warning(f"⚠️ Column {col} not found, skipping")
                continue
                
            try:
                if self.harmonization_type == "direction":
                    # Convert to signed direction [-1, 1]
                    # Assuming predictions are probabilities [0,1]
                    pred_values = predictions_df[col].values
                    harmonized[f"{col}_harmonized"] = 2 * pred_values - 1
                    
                elif self.harmonization_type == "probability":
                    # Ensure probabilities are in [0,1]
                    pred_values = predictions_df[col].values
                    harmonized[f"{col}_harmonized"] = np.clip(pred_values, 0.0, 1.0)
                    
                else:
                    raise ValueError(f"Unknown harmonization type: {self.harmonization_type}")
                    
            except Exception as e:
                tprint_warning(f"⚠️ Failed to harmonize {col}: {e}")
                harmonized[f"{col}_harmonized"] = 0.0  # Fallback
        
        # Validate harmonization
        harmonized_stats = harmonized.describe()
        tprint_info(f"📊 Harmonization Statistics:")
        for col in [c for c in harmonized.columns if "_harmonized" in c]:
            col_stats = harmonized_stats[col]
            tprint_info(f"   {col}: mean={col_stats['mean']:.4f}, std={col_stats['std']:.4f}, min={col_stats['min']:.4f}, max={col_stats['max']:.4f}")
        
        self.harmonized_predictions_ = harmonized
        return harmonized
    
    def generate_contextual_residual_features(
        self,
        predictions_df: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """
        Step 2: Generate contextual residual features from harmonized predictions.
        
        Args:
            predictions_df: DataFrame with harmonized predictions
            target_col: Target column name
            
        Returns:
            DataFrame with 60 contextual residual features (20 models × 3 contexts)
        """
        tprint_info("🔧 Step 2: Generating Contextual Residual Features...")
        
        if self.harmonized_predictions_ is None:
            tprint_error("❌ No harmonized predictions found. Run harmonization first.")
            raise ValueError("No harmonized predictions found")
        
        # Get harmonized columns
        harmonized_cols = [col for col in self.harmonized_predictions_.columns if "_harmonized" in col]
        base_models = [col.replace("_harmonized", "") for col in harmonized_cols]
        
        tprint_info(f"📊 Generating residual features for {len(base_models)} base models")
        
        # Extract target values
        y_true = self.harmonized_predictions_[target_col].values
        if self.harmonization_type == "direction":
            # Convert target to direction [-1, 1]
            y_true = np.where(y_true > 0.5, 1.0, -1.0)
        else:
            # Keep as probability [0, 1]
            y_true = y_true
        
        residual_features = pd.DataFrame(index=self.harmonized_predictions_.index)
        
        for i, (base_model, harmonized_col) in enumerate(zip(base_models, harmonized_cols)):
            try:
                y_pred = self.harmonized_predictions_[harmonized_col].values
                
                # Calculate residuals
                residual = y_true - y_pred
                residual_features[f"{base_model}_residual"] = residual
                
                # Calculate bias (EMA of residuals)
                bias_ema = pd.Series(residual).ewm(span=self.bias_window, adjust=False).mean()
                residual_features[f"{base_model}_bias_ema"] = bias_ema
                
                # Calculate volatility (rolling std of residuals)
                residual_vol = pd.Series(residual).rolling(window=self.volatility_window, min_periods=1).std()
                residual_features[f"{base_model}_volatility"] = residual_vol
                
                # Calculate reliability (CUSUM of absolute residuals)
                abs_residual = np.abs(residual)
                cusum_pos = np.zeros(len(abs_residual))
                cusum_neg = np.zeros(len(abs_residual))
                cusum_signal = np.zeros(len(abs_residual))
                
                for t in range(1, len(abs_residual)):
                    cusum_pos[t] = max(0.0, cusum_pos[t-1] + abs_residual[t])
                    cusum_neg = min(0.0, cusum_neg[t-1] - abs_residual[t])
                    
                    if cusum_pos[t] > self.cusum_threshold or abs(cusum_neg[t]) > self.cusum_threshold:
                        cusum_signal[t] = 1 if cusum_pos[t] > self.cusum_threshold else -1
                        cusum_pos[t] = 0.0
                        cusum_neg[t] = 0.0
                
                residual_features[f"{base_model}_reliability_cusum"] = cusum_signal
                
                # Progress tracking
                if (i + 1) % 5 == 0 or i == len(base_models) - 1:
                    progress = (i + 1) / len(base_models) * 100
                    tprint_info(f"   Progress: {i+1}/{len(base_models)} ({progress:.1f}%) - {base_model}")
                
            except Exception as e:
                tprint_warning(f"⚠️ Failed to generate residual features for {base_model}: {e}")
                # Fill with zeros for failed models
                residual_features[f"{base_model}_residual"] = 0.0
                residual_features[f"{base_model}_bias_ema"] = 0.0
                residual_features[f"{base_model}_volatility"] = 0.0
                residual_features[f"{base_model}_reliability_cusum"] = 0.0
        
        # Calculate feature statistics
        self._calculate_residual_statistics(residual_features)
        
        self.residual_features_ = residual_features
        
        # Enhanced detailed reporting
        if cfg.get("detailed_residual_reporting", True):
            self._print_detailed_residual_report()
        return residual_features
    
    def _calculate_residual_statistics(self, residual_features: pd.DataFrame) -> None:
        """Calculate and store statistics for residual features."""
        tprint_info("📊 Calculating Residual Feature Statistics...")
        
        stats = {}
        for col in residual_features.columns:
            if col.endswith('_residual'):
                # Residual statistics
                stats[col] = {
                    'mean': residual_features[col].mean(),
                    'std': residual_features[col].std(),
                    'skew': residual_features[col].skew(),
                    'kurtosis': residual_features[col].kurtosis(),
                    'non_zero_pct': (residual_features[col] != 0).mean() * 100
                }
            elif col.endswith('_bias_ema'):
                # Bias statistics
                stats[col] = {
                    'mean': residual_features[col].mean(),
                    'std': residual_features[col].std(),
                    'range': residual_features[col].max() - residual_features[col].min(),
                    'positive_pct': (residual_features[col] > 0).mean() * 100
                }
            elif col.endswith('_volatility'):
                # Volatility statistics
                stats[col] = {
                    'mean': residual_features[col].mean(),
                    'std': residual_features[col].std(),
                    'min': residual_features[col].min(),
                    'max': residual_features[col].max(),
                    'stability': 1.0 / (residual_features[col].std() + EPS)
                }
            elif col.endswith('_reliability_cusum'):
                # Reliability statistics
                stats[col] = {
                    'signal_count': (residual_features[col] != 0).sum(),
                    'signal_rate': (residual_features[col] != 0).mean() * 100,
                    'mean': residual_features[col].mean(),
                    'std': residual_features[col].std()
                }
        
        self.feature_stats_ = stats
        
        # Report summary statistics
        residual_count = len([col for col in residual_features.columns if col.endswith('_residual')])
        bias_count = len([col for col in residual_features.columns if col.endswith('_bias_ema')])
        vol_count = len([col for col in residual_features.columns if col.endswith('_volatility')])
        reliability_count = len([col for col in residual_features.columns if col.endswith('_reliability_cusum')])
        
        tprint_success(f"✅ Generated {len(residual_features.columns)} residual features:")
        tprint_info(f"   📊 Residual features: {residual_count}")
        tprint_info(f"   📊 Bias features: {bias_count}")
        tprint_info(f"   📊 Volatility features: {vol_count}")
        tprint_info(f"   📊 Reliability features: {reliability_count}")
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all residual features.
        
        Returns:
            Dictionary with feature statistics
        """
        if self.feature_stats_ is None:
            raise ValueError("No statistics available. Run generate_contextual_residual_features() first.")
        return self.feature_stats_.copy()
    
    def get_harmonized_predictions(self) -> pd.DataFrame:
        """
        Get harmonized predictions.
        
        Returns:
            DataFrame with harmonized predictions
        """
        if self.harmonized_predictions_ is None:
            raise ValueError("No harmonized predictions available. Run harmonize_base_model_predictions() first.")
        return self.harmonized_predictions_.copy()
    
    def get_residual_features(self) -> pd.DataFrame:
        """
        Get residual features.
        
        Returns:
            DataFrame with residual features
        """
        if self.residual_features_ is None:
            raise ValueError("No residual features available. Run generate_contextual_residual_features() first.")
        return self.residual_features_.copy()


def harmonize_base_model_predictions(
    predictions_df: pd.DataFrame,
    target_col: str,
    harmonization_type: str = "direction",
    min_samples: int = 100
) -> pd.DataFrame:
    """
    Convenience function for harmonizing base model predictions.
    
    Args:
        predictions_df: DataFrame with base model predictions
        target_col: Target column name
        harmonization_type: Type of harmonization ("direction" or "probability")
        min_samples: Minimum samples required
        
    Returns:
        DataFrame with harmonized predictions
    """
    generator = ContextualResidualFeatureGenerator(
        harmonization_type=harmonization_type,
        min_samples=min_samples
    )
    
    return generator.harmonize_base_model_predictions(predictions_df, target_col)


def generate_contextual_residual_features(
    predictions_df: pd.DataFrame,
    target_col: str,
    harmonization_type: str = "direction",
    bias_window: int = 20,
    volatility_window: int = 30,
    reliability_window: int = 50,
    cusum_threshold: float = 2.0,
    min_samples: int = 100
) -> Tuple[pd.DataFrame, ContextualResidualFeatureGenerator]:
    """
    Convenience function for generating contextual residual features.
    
    Args:
        predictions_df: DataFrame with harmonized predictions
        target_col: Target column name
        harmonization_type: Type of harmonization
        bias_window: Window size for EMA bias calculation
        volatility_window: Window size for rolling volatility
        reliability_window: Window size for CUSUM reliability
        cusum_threshold: Threshold for CUSUM signal generation
        min_samples: Minimum samples required
        
    Returns:
        Tuple of (residual_features_df, fitted_generator)
    """
    generator = ContextualResidualFeatureGenerator(
        harmonization_type=harmonization_type,
        bias_window=bias_window,
        volatility_window=volatility_window,
        reliability_window=reliability_window,
        cusum_threshold=cusum_threshold,
        min_samples=min_samples
    )
    
    # First harmonize if needed
    if generator.harmonized_predictions_ is None:
        generator.harmonize_base_model_predictions(predictions_df, target_col)
    
    # Then generate residual features
    residual_features = generator.generate_contextual_residual_features(predictions_df, target_col)
    
    return residual_features, generator

    def _print_detailed_residual_report(self) -> None:
        """
        Print detailed contextual residual feature report with model-by-model analysis.
        """
        if self.feature_stats_ is None:
            tprint_warning("⚠️ No residual statistics available for detailed reporting")
            return
        
        max_display = 20  # Limit output for large feature sets
        
        tprint_info("🔍 Contextual Residual Feature Report:")
        tprint_info(f"📊 Generated {len(self.feature_stats_)} residual features from {len([c for c in self.feature_stats_.keys() if c.endswith('_residual')])} base models")
        
        # Group features by type
        residual_features = {k: v for k, v in self.feature_stats_.items() if k.endswith('_residual')}
        bias_features = {k: v for k, v in self.feature_stats_.items() if k.endswith('_bias_ema')}
        volatility_features = {k: v for k, v in self.feature_stats_.items() if k.endswith('_volatility')}
        reliability_features = {k: v for k, v in self.feature_stats_.items() if k.endswith('_reliability_cusum')}
        
        # Extract base model names
        base_models = list(set([k.replace('_residual', '') for k in residual_features.keys()]))
        
        # Model-by-model analysis
        tprint_info(f"🤖 Model-by-Model Performance Analysis:")
        
        for model in base_models[:max_display]:  # Limit display
            model_residual = residual_features.get(f"{model}_residual", {})
            model_bias = bias_features.get(f"{model}_bias_ema", {})
            model_volatility = volatility_features.get(f"{model}_volatility", {})
            model_reliability = reliability_features.get(f"{model}_reliability_cusum", {})
            
            # Determine model performance status
            residual_mean = model_residual.get('mean', 0)
            bias_mean = model_bias.get('mean', 0)
            volatility_mean = model_volatility.get('mean', 0)
            reliability_signal_rate = model_reliability.get('signal_rate', 0)
            
            # Performance classification
            if abs(residual_mean) < 0.1 and reliability_signal_rate < 5:
                status = "🟢 EXCELLENT"
                reason = "Low bias, stable performance"
            elif abs(residual_mean) < 0.2 and reliability_signal_rate < 10:
                status = "🟡 GOOD"
                reason = "Moderate bias, acceptable stability"
            elif abs(residual_mean) < 0.5:
                status = "🟠 FAIR"
                reason = "High bias, some instability"
            else:
                status = "🔴 POOR"
                reason = "Very high bias, unstable performance"
            
            tprint_info(f"   {status} {model}: {reason}")
            tprint_info(f"      📊 Residual mean: {residual_mean:.4f}, Bias: {bias_mean:.4f}")
            tprint_info(f"      📊 Volatility: {volatility_mean:.4f}, Reliability signals: {reliability_signal_rate:.1f}%")
        
        if len(base_models) > max_display:
            tprint_info(f"   ... and {len(base_models) - max_display} more models")
        
        # Feature type analysis
        tprint_info(f"📊 Feature Type Analysis:")
        
        # Best performing residual features (lowest absolute mean)
        best_residuals = sorted(residual_features.items(), key=lambda x: abs(x[1]['mean']))[:5]
        tprint_info(f"   🏆 Most Accurate Models (lowest residual bias):")
        for feature, stats in best_residuals:
            model = feature.replace('_residual', '')
            tprint_info(f"      {model}: bias={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        # Most stable models (lowest volatility)
        most_stable = sorted(volatility_features.items(), key=lambda x: x[1]['mean'])[:5]
        tprint_info(f"   🛡️  Most Stable Models (lowest volatility):")
        for feature, stats in most_stable:
            model = feature.replace('_volatility', '')
            tprint_info(f"      {model}: volatility={stats['mean']:.4f}, stability={stats['stability']:.2f}")
        
        # Most reliable models (fewest structural breaks)
        most_reliable = sorted(reliability_features.items(), key=lambda x: x[1]['signal_rate'])[:5]
        tprint_info(f"   🔒 Most Reliable Models (fewest structural breaks):")
        for feature, stats in most_reliable:
            model = feature.replace('_reliability_cusum', '')
            tprint_info(f"      {model}: signal_rate={stats['signal_rate']:.1f}%, signals={stats['signal_count']}")
        
        # Problematic models
        problematic_residuals = sorted(residual_features.items(), key=lambda x: abs(x[1]['mean']), reverse=True)[:3]
        tprint_info(f"   ⚠️  Most Problematic Models (highest residual bias):")
        for feature, stats in problematic_residuals:
            model = feature.replace('_residual', '')
            tprint_info(f"      {model}: bias={stats['mean']:.4f}, non_zero={stats['non_zero_pct']:.1f}%")
        
        # Summary statistics
        tprint_info(f"📊 Residual Feature Summary:")
        tprint_info(f"   📈 Total residual features: {len(self.feature_stats_)}")
        tprint_info(f"   🤖 Base models analyzed: {len(base_models)}")
        tprint_info(f"   📊 Feature types: {len(residual_features)} residual, {len(bias_features)} bias, {len(volatility_features)} volatility, {len(reliability_features)} reliability")
        
        # Overall model health
        avg_residual_bias = np.mean([abs(stats['mean']) for stats in residual_features.values()])
        avg_volatility = np.mean([stats['mean'] for stats in volatility_features.values()])
        avg_reliability_rate = np.mean([stats['signal_rate'] for stats in reliability_features.values()])
        
        tprint_info(f"   🏥 Overall Model Health:")
        tprint_info(f"      📊 Average bias: {avg_residual_bias:.4f}")
        tprint_info(f"      📊 Average volatility: {avg_volatility:.4f}")
        tprint_info(f"      📊 Average structural break rate: {avg_reliability_rate:.1f}%")
        
        if avg_residual_bias < 0.2 and avg_reliability_rate < 10:
            tprint_info(f"      🟢 Overall model ensemble health: EXCELLENT")
        elif avg_residual_bias < 0.4 and avg_reliability_rate < 20:
            tprint_info(f"      🟡 Overall model ensemble health: GOOD")
        else:
            tprint_info(f"      🔴 Overall model ensemble health: NEEDS ATTENTION")
