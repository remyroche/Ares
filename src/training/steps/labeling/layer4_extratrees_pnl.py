"""
Layer 4 — ExtraTrees Position Sizing Model

Layer4 redesigned to use ExtraTrees classifier trained on returns to maximize PnL 
and Sortino while minimizing drawdown.

Features:
1. OOF predictions from layer3
2. Disagreement features from ensemble_disagreement.py
3. Average of heads ProbA * ProbB from layer3
4. Past Precision (rolling accuracy in similar market conditions)
5. Structural Break Scores (SADF/CUSUM filter values)
6. Relative Strength (sector-relative performance or VWAP distance)
7. Drawdown State (peak/trough detection)
"""

import numpy as np
import time
import time
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from datetime import datetime
import json

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# Import ensemble disagreement features
from src.feature_generation.categories.ensemble_disagreement import EnsembleDisagreementFeatures
# Import advanced feature selection utilities
from src.training.steps.labeling.conditional_mutual_information import ConditionalMutualInformationSelector, cmi_feature_selection
from src.training.steps.labeling.contextual_residual_features import ContextualResidualFeatureGenerator, generate_contextual_residual_features
from src.training.steps.labeling.de_prado_feature_engine import DePradoFeatureEngine, de_prado_feature_selection


def calculate_structural_break_scores(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Calculate structural break scores using SADF and CUSUM filters.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        
    Returns:
        DataFrame with structural break scores
    """
    try:
        prices = df[price_col].values
        returns = np.diff(np.log(prices + 1e-8))
        
        # SADF (Supremum Augmented Dickey-Fuller) - bubble detection
        sadf_scores = []
        window_size = min(100, len(returns) // 4)
        
        for i in range(window_size, len(returns)):
            window_returns = returns[i-window_size:i]
            
            # Simple ADF-like statistic (simplified)
            if len(window_returns) > 10:
                # Calculate test statistic
                x = window_returns[:-1]
                y = window_returns[1:]
                
                # Simple regression y = alpha + beta*x
                if len(x) > 1 and np.std(x) > 1e-8:
                    beta = np.cov(x, y)[0, 1] / np.var(x)
                    alpha = np.mean(y) - beta * np.mean(x)
                    residuals = y - (alpha + beta * x)
                    
                    # Test statistic (simplified ADF)
                    if np.std(residuals) > 1e-8:
                        t_stat = beta / (np.std(residuals) / np.sqrt(len(x) * np.var(x)))
                        sadf_scores.append(abs(t_stat))
                    else:
                        sadf_scores.append(0.0)
                else:
                    sadf_scores.append(0.0)
            else:
                sadf_scores.append(0.0)
        
        # CUSUM filter - change point detection
        cusum_scores = []
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 1e-8:
            cusum_pos = 0
            cusum_neg = 0
            
            for ret in returns:
                cusum_pos = max(0, cusum_pos + (ret - mean_return))
                cusum_neg = min(0, cusum_neg + (ret - mean_return))
                cusum_scores.append(abs(cusum_pos) + abs(cusum_neg))
        else:
            cusum_scores = [0.0] * len(returns)
        
        # Align with original DataFrame
        result_df = df.copy()
        result_df['sadf_score'] = np.nan
        result_df['cusum_score'] = np.nan
        
        # Fill with calculated scores
        result_df.iloc[window_size+1:, result_df.columns.get_loc('sadf_score')] = sadf_scores
        result_df.iloc[1:, result_df.columns.get_loc('cusum_score')] = cusum_scores[:len(df)-1]
        
        # Fill NaNs
        result_df['sadf_score'] = result_df['sadf_score'].fillna(0.0)
        result_df['cusum_score'] = result_df['cusum_score'].fillna(0.0)
        
        # Normalize scores
        result_df['sadf_score_norm'] = result_df['sadf_score'] / (result_df['sadf_score'].max() + 1e-8)
        result_df['cusum_score_norm'] = result_df['cusum_score'] / (result_df['cusum_score'].max() + 1e-8)
        
        return result_df
        
    except Exception as e:
        tprint_warning(f"Error calculating structural break scores: {e}")
        result_df = df.copy()
        result_df['sadf_score_norm'] = 0.0
        result_df['cusum_score_norm'] = 0.0
        return result_df


def calculate_past_precision(
    df: pd.DataFrame, 
    target_col: str = 'realized_return',
    prob_col: str = 'meta_prob',
    window: int = 50
) -> pd.Series:
    """
    Calculate past precision (rolling accuracy) in similar market conditions.
    
    Args:
        df: DataFrame with predictions and targets
        target_col: Target column name
        prob_col: Probability column name  
        window: Rolling window size
        
    Returns:
        Series with past precision scores
    """
    try:
        # Convert probabilities to binary predictions
        predictions = (df[prob_col] > 0.5).astype(int)
        targets = (df[target_col] > 0).astype(int)
        
        # Rolling accuracy
        rolling_correct = predictions.rolling(window).sum()
        rolling_total = pd.Series(1, index=df.index).rolling(window).sum()
        
        past_precision = rolling_correct / rolling_total
        
        return past_precision.fillna(0.5)
        
    except Exception as e:
        tprint_warning(f"Error calculating past precision: {e}")
        return pd.Series(0.5, index=df.index)


def calculate_relative_strength(
    df: pd.DataFrame,
    price_col: str = 'close',
    volume_col: str = 'volume',
    window: int = 20
) -> pd.DataFrame:
    """
    Calculate relative strength metrics (sector-relative performance, VWAP distance).
    
    Args:
        df: DataFrame with price and volume data
        price_col: Price column name
        volume_col: Volume column name
        window: Rolling window size
        
    Returns:
        DataFrame with relative strength features
    """
    try:
        result_df = df.copy()
        
        # VWAP calculation
        if volume_col in df.columns:
            typical_price = (df['high'] + df['low'] + 2 * df[price_col]) / 4
            vwap = (typical_price * df[volume_col]).rolling(window).sum() / df[volume_col].rolling(window).sum()
            
            # Distance from VWAP
            result_df['vwap_distance'] = (df[price_col] - vwap) / vwap
            result_df['vwap_ratio'] = df[price_col] / (vwap + 1e-8)
        else:
            result_df['vwap_distance'] = 0.0
            result_df['vwap_ratio'] = 1.0
        
        # Relative strength vs moving average
        ma = df[price_col].rolling(window).mean()
        result_df['relative_strength_ma'] = (df[price_col] - ma) / (ma + 1e-8)
        
        # Momentum vs recent average
        short_ma = df[price_col].rolling(window//2).mean()
        result_df['relative_strength_short'] = (df[price_col] - short_ma) / (short_ma + 1e-8)
        
        return result_df
        
    except Exception as e:
        tprint_warning(f"Error calculating relative strength: {e}")
        result_df = df.copy()
        result_df['vwap_distance'] = 0.0
        result_df['vwap_ratio'] = 1.0
        result_df['relative_strength_ma'] = 0.0
        result_df['relative_strength_short'] = 0.0
        return result_df


def calculate_drawdown_state(
    df: pd.DataFrame,
    price_col: str = 'close',
    window: int = 50
) -> pd.DataFrame:
    """
    Calculate drawdown state (peak/trough detection).
    
    Args:
        df: DataFrame with price data
        price_col: Price column name
        window: Rolling window size
        
    Returns:
        DataFrame with drawdown state features
    """
    try:
        result_df = df.copy()
        
        # Calculate rolling peak and trough
        rolling_peak = df[price_col].rolling(window).max()
        rolling_trough = df[price_col].rolling(window).min()
        
        # Current drawdown from peak
        result_df['drawdown_from_peak'] = (df[price_col] - rolling_peak) / rolling_peak
        
        # Distance from trough (recovery potential)
        result_df['distance_from_trough'] = (df[price_col] - rolling_trough) / (rolling_trough + 1e-8)
        
        # Is currently near peak (> 90% of rolling peak)
        result_df['is_near_peak'] = (df[price_col] > 0.9 * rolling_peak).astype(float)
        
        # Is currently near trough (< 110% of rolling trough)
        result_df['is_near_trough'] = (df[price_col] < 1.1 * rolling_trough).astype(float)
        
        # Drawdown regime classification
        drawdown = result_df['drawdown_from_peak']
        result_df['drawdown_regime_severe'] = (drawdown < -0.1).astype(float)  # >10% drawdown
        result_df['drawdown_regime_moderate'] = ((drawdown >= -0.1) & (drawdown < -0.05)).astype(float)  # 5-10% drawdown
        result_df['drawdown_regime_mild'] = ((drawdown >= -0.05) & (drawdown < -0.02)).astype(float)  # 2-5% drawdown
        result_df['drawdown_regime_none'] = (drawdown >= -0.02).astype(float)  # <2% drawdown
        
        return result_df
        
    except Exception as e:
        tprint_warning(f"Error calculating drawdown state: {e}")
        result_df = df.copy()
        result_df['drawdown_from_peak'] = 0.0
        result_df['distance_from_trough'] = 0.0
        result_df['is_near_peak'] = 0.0
        result_df['is_near_trough'] = 0.0
        result_df['drawdown_regime_severe'] = 0.0
        result_df['drawdown_regime_moderate'] = 0.0
        result_df['drawdown_regime_mild'] = 0.0
        result_df['drawdown_regime_none'] = 1.0
        return result_df


def generate_layer4_features(
    df: pd.DataFrame,
    layer3_predictions: pd.DataFrame,
    target_col: str = 'realized_return',
    prob_col: str = 'meta_prob',
    use_raw_returns: bool = True,
    use_weights: bool = True
) -> pd.DataFrame:
    """
    Generate comprehensive Layer4 features for ExtraTrees model.
    
    Args:
        df: Market data DataFrame
        layer3_predictions: Layer3 OOF predictions DataFrame
        target_col: Target column name
        prob_col: Layer3 probability column name
        use_raw_returns: Whether to use raw returns (vs denoised)
        use_weights: Whether to use sample weights
        
    Returns:
        DataFrame with Layer4 features
    """
    try:
        # Deduplicate inputs first to prevent length mismatch
        if df.index.has_duplicates:
            tprint_warning(f"⚠️ Layer 4 input 'df' has duplicates: {df.index.duplicated().sum()}")
            df = df[~df.index.duplicated(keep='first')]
            
        if layer3_predictions.index.has_duplicates:
            tprint_warning(f"⚠️ Layer 4 input 'layer3_predictions' has duplicates: {layer3_predictions.index.duplicated().sum()}")
            layer3_predictions = layer3_predictions[~layer3_predictions.index.duplicated(keep='first')]

        # Combine data with suffixes to avoid column overlap
        combined_df = df.join(layer3_predictions, how='inner', rsuffix='_l3')
        
        # 1. OOF predictions from layer3
        layer3_prob_cols = [c for c in combined_df.columns if c.startswith('meta_prob_') or c == prob_col]
        
        # 2. Disagreement features
        disagreement_features = []
        if 'ensemble_disagreement' in combined_df.columns:
            # If disagreement features already calculated
            disagreement_cols = [c for c in combined_df.columns if any(
                feature in c for feature in ['prediction_dispersion', 'confidence_gap', 'uncertainty', 
                                           'prediction_range', 'avg_divergence', 'max_confidence',
                                           'disagreement_rate', 'snr_internal', 'snr_consensus']
            )]
            disagreement_features = disagreement_cols
        else:
            # Calculate disagreement features if we have multiple model predictions
            model_cols = [c for c in combined_df.columns if c.startswith('model_')]
            if len(model_cols) > 1:
                calculator = EnsembleDisagreementFeatures()
                model_predictions = {col: combined_df[col].values for col in model_cols}
                model_probabilities = {col: combined_df[col].values for col in model_cols}
                
                disagreement_df = calculator.calculate_disagreement_features(
                    model_predictions, model_probabilities
                )
                combined_df = pd.concat([combined_df, disagreement_df], axis=1)
                disagreement_features = list(disagreement_df.columns)
        
        # 3. Average of heads ProbA * ProbB from layer3
        if len(layer3_prob_cols) >= 2:
            prob_matrix = combined_df[layer3_prob_cols].values
            # Calculate pairwise products and average
            n_models = len(layer3_prob_cols)
            pairwise_products = []
            for i in range(n_models):
                for j in range(i+1, n_models):
                    pairwise_products.append(prob_matrix[:, i] * prob_matrix[:, j])
            combined_df['avg_prob_product'] = np.mean(pairwise_products, axis=0)
        else:
            combined_df['avg_prob_product'] = combined_df[prob_col]
        
        # 4. Past Precision
        past_precision = calculate_past_precision(combined_df, target_col, prob_col)
        combined_df['past_precision'] = past_precision
        
        # 5. Structural Break Scores
        structural_features = calculate_structural_break_scores(combined_df)
        combined_df['sadf_score_norm'] = structural_features['sadf_score_norm']
        combined_df['cusum_score_norm'] = structural_features['cusum_score_norm']
        
        # 6. Relative Strength (simplified)
        relative_strength = calculate_relative_strength(combined_df)
        # DISABLED: VWAP and relative strength features
        # combined_df['vwap_distance'] = relative_strength['vwap_distance']
        # combined_df['vwap_ratio'] = relative_strength['vwap_ratio']
        # combined_df['relative_strength_ma'] = relative_strength['relative_strength_ma']
        # combined_df['relative_strength_short'] = relative_strength['relative_strength_short']
        
        # 7. Drawdown State (DISABLED)
        # DISABLED: Drawdown features
        # drawdown_features = calculate_drawdown_state(combined_df)
        # drawdown_cols = ['drawdown_from_peak', 'distance_from_trough', 'is_near_peak', 'is_near_trough',
        #                 'drawdown_regime_severe', 'drawdown_regime_moderate', 'drawdown_regime_mild', 'drawdown_regime_none']
        # for col in drawdown_cols:
        #     combined_df[col] = drawdown_features[col]
        
        # 8. Additional market features (simplified)
        if 'volatility_1d' in combined_df.columns:
            vol = combined_df['volatility_1d']
            # DISABLED: Volatility features
            # combined_df['volatility_zscore'] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-8)
            # combined_df['volatility_regime'] = (vol > vol.rolling(50).mean()).astype(float)
        
        # 9. Time features (simplified - only hour_sin/cos from regime features)
        # DISABLED: Additional time features
        # if hasattr(combined_df.index, 'hour'):
        #     combined_df['hour_of_day'] = combined_df.index.hour
        #     combined_df['day_of_week'] = combined_df.index.dayofweek
        #     combined_df['is_session_start'] = ((combined_df.index.hour >= 8) & (combined_df.index.hour <= 10)).astype(float)
        #     combined_df['is_session_end'] = ((combined_df.index.hour >= 16) & (combined_df.index.hour <= 18)).astype(float)
        
        # Clean up infinite and NaN values
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            combined_df[col] = combined_df[col].replace([np.inf, -np.inf], np.nan)
            combined_df[col] = combined_df[col].fillna(combined_df[col].rolling(20, min_periods=1).mean())
            combined_df[col] = combined_df[col].fillna(0.0)
        
        return combined_df
        
    except Exception as e:
        tprint_error(f"Error generating Layer4 features: {e}")
        return df.join(layer3_predictions, how='inner', rsuffix='_l3')


def train_layer4_extratrees(
    df: pd.DataFrame,
    layer3_predictions: pd.DataFrame,
    target_col: str = 'realized_return',
    prob_col: str = 'meta_prob',
    use_raw_returns: bool = True,
    use_weights: bool = True,
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train ExtraTrees model for Layer4 position sizing focused on PnL and Sortino optimization.
    
    Args:
        df: Market data DataFrame
        layer3_predictions: Layer3 OOF predictions DataFrame
        target_col: Target column name
        prob_col: Layer3 probability column name
        use_raw_returns: Whether to use raw returns (vs denoised)
        use_weights: Whether to use sample weights in training
        n_folds: Number of CV folds
        config: Configuration dictionary
        
    Returns:
        Tuple of (predictions DataFrame, metadata dictionary)
    """
    
    tprint_info("="*60)
    tprint_info("🚀 LAYER 4: EXTRATREES POSITION SIZING (PnL & SORTINO OPTIMIZED)")
    tprint_info("="*60)
    
    cfg = config or {}
    
    tprint_info(f"📊 Input data: {len(df)} market samples, {len(layer3_predictions)} Layer3 predictions")
    tprint_info(f"🎯 Target: {target_col}, Using raw returns: {use_raw_returns}, Using weights: {use_weights}")
    
    # 1. Generate features
    tprint_info("🔧 Generating Layer4 features...")
    df_features = generate_layer4_features(
        df, layer3_predictions, target_col, prob_col, use_raw_returns, use_weights
    )
    
    # 2. Prepare target variable
    if target_col not in df_features.columns:
        tprint_error(f"❌ Target column '{target_col}' not found")
        return df_features, {}
    
    # Create binary target for classification (positive vs negative returns)
    returns = pd.to_numeric(df_features[target_col], errors='coerce').dropna()
    binary_target = (returns > 0).astype(int)
    
    # Align data
    common_idx = returns.index.intersection(df_features.index)
    df_aligned = df_features.loc[common_idx]
    returns_aligned = returns.loc[common_idx]
    binary_target_aligned = binary_target.loc[common_idx]
    
    # 3. Feature selection
    # Select relevant features for ExtraTrees
    feature_candidates = []
    
    # Layer3 probability features
    layer3_cols = [c for c in df_aligned.columns if c.startswith('meta_prob_') or c == prob_col]
    feature_candidates.extend(layer3_cols)
    
    # Disagreement features
    disagreement_cols = [c for c in df_aligned.columns if any(
        feature in c for feature in ['prediction_dispersion', 'confidence_gap', 'uncertainty', 
                                   'prediction_range', 'avg_divergence', 'max_confidence',
                                   'disagreement_rate', 'snr_internal', 'snr_consensus']
    )]
    feature_candidates.extend(disagreement_cols)
    
    # Core Layer4 features
    core_features = [
        'avg_prob_product', 'past_precision', 'sadf_score_norm', 'cusum_score_norm',
        'vwap_distance', 'vwap_ratio', 'relative_strength_ma', 'relative_strength_short',
        'drawdown_from_peak', 'distance_from_trough', 'is_near_peak', 'is_near_trough',
        'drawdown_regime_severe', 'drawdown_regime_moderate', 'drawdown_regime_mild', 'drawdown_regime_none',
        'volatility_zscore', 'volatility_regime', 'zone_score', 'zone3_ratio', 'zone2_ratio'
    ]
    feature_candidates.extend([f for f in core_features if f in df_aligned.columns])
    
    # Time features
    time_features = ['hour_of_day', 'day_of_week', 'is_session_start', 'is_session_end']
    feature_candidates.extend([f for f in time_features if f in df_aligned.columns])
    
    # Causal features (from Layer 2 causal targets)
    causal_features = [
        'causal_effect_estimate', 'causal_effect_ci_low', 'causal_effect_ci_high',
        'causal_refutation_score', 'causal_residuals', 'cate_estimates',
        'heterogeneity_score', 'treatment_residuals', 'causal_bet_size',
        'causal_confidence', 'causal_residual_zscore', 'causal_residual_momentum',
        'causal_reliability', 'causal_validity', 'cate_strength', 'cate_direction',
        'cate_volatility', 'heterogeneity_magnitude', 'heterogeneity_regime'
    ]
    feature_candidates.extend([f for f in causal_features if f in df_aligned.columns])
    
    # Remove any remaining non-numeric or target columns
    available_features = []
    for f in feature_candidates:
        if f in df_aligned.columns and f != target_col:
            if pd.api.types.is_numeric_dtype(df_aligned[f]):
                available_features.append(f)
    
    # ---------------------------------------------------------
    # Advanced Feature Selection (CMI + De Prado)
    # ---------------------------------------------------------
    
    # Check if advanced feature selection is enabled (default: True)
        # ---------------------------------------------------------
        # Contextual Residual Feature Generation (NEW)
        # ---------------------------------------------------------
        
        enable_residual_features = config.get("enable_residual_features", True) if config else True
        
        if enable_residual_features and len(layer3_prob_cols) > 3:
            tprint_info("🔍 Generating Contextual Residual Features for Layer 4...")
            
            try:
                residual_start = time.time()
                
                # Create predictions DataFrame for residual analysis
                predictions_df = df_aligned[layer3_prob_cols + [prob_col]].copy()
                predictions_df = predictions_df.rename(columns={prob_col: "target"})
                
                # Generate contextual residual features
                residual_features, residual_generator = generate_contextual_residual_features(
                    predictions_df=predictions_df,
                    target_col="target",
                    harmonization_type=config.get("harmonization_type", "direction") if config else "direction",
                    bias_window=config.get("bias_window", 20) if config else 20,
                    volatility_window=config.get("volatility_window", 30) if config else 30,
                    reliability_window=config.get("reliability_window", 50) if config else 50,
                    cusum_threshold=config.get("cusum_threshold", 2.0) if config else 2.0
                )
                
                # Add residual features to aligned dataframe
                for col in residual_features.columns:
                    df_aligned[col] = residual_features[col].reindex(df_aligned.index).fillna(0.0)
                    available_features.append(col)
                
                residual_time = time.time() - residual_start
                tprint_success(f"✅ Generated {len(residual_features.columns)} residual features for Layer 4 in {residual_time:.2f}s")
                tprint_info(f"📊 Layer 4 residual feature count: {len(available_features)} total")
                
                # Save residual feature reports for Layer 4
                try:
                    outcomes_dir = Path("outcomes")
                    outcomes_dir.mkdir(exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    residual_features.to_csv(outcomes_dir / f"layer4_residual_features_{ts}.csv")
                    residual_generator.get_feature_statistics()
                    tprint_info(f"📁 Saved Layer 4 residual reports")
                except Exception as e:
                    tprint_warning(f"   Failed to save Layer 4 residual reports: {e}")
                    
            except Exception as e:
                tprint_warning(f"   ⚠️ Layer 4 residual feature generation failed: {e}")
                tprint_info(f"   Continuing without residual features...")
        
    enable_advanced_selection = config.get("enable_advanced_feature_selection", True) if config else True
    
    if enable_advanced_selection and len(available_features) > 15:
        advanced_start = time.time()
        tprint_info("🔍 Running Advanced Feature Selection (CMI + De Prado) for Layer 4...")
        tprint_info(f"📊 Layer 4: Starting with {len(available_features)} features")
        tprint_info("🔍 Running Advanced Feature Selection (CMI + De Prado) for Layer 4...")
        
        try:
            # Create feature matrix for selection
            X_raw = df_aligned[available_features].fillna(0)
            y_raw = binary_target_aligned
            
            tprint_info(f"📊 Layer 4 feature matrix: {X_raw.shape}")
            tprint_info(f"📊 Target distribution: {y_raw.mean():.3f} positive, {1-y_raw.mean():.3f} negative")
            
            # Check for any issues with features
            zero_var_features = X_raw.var() == 0
            if zero_var_features.any():
                n_zero_var = zero_var_features.sum()
                tprint_warning(f"⚠️ Found {n_zero_var} zero-variance features")
            
            # Get base predictions for CMI (use main layer3 probability)
            if prob_col in df_aligned.columns:
                base_predictions = df_aligned[prob_col]
                tprint_info(f"📊 Using main layer3 probability: {prob_col}")
            else:
                base_predictions = df_aligned[layer3_prob_cols[0]]
                tprint_info(f"📊 Using fallback layer3 probability: {layer3_prob_cols[0]}")
            
            # Data quality check
            if base_predictions.isnull().any():
                n_missing = base_predictions.isnull().sum()
                tprint_warning(f"⚠️ Filling {n_missing} missing base predictions with 0.5")
                base_predictions = base_predictions.fillna(0.5)
            
            # Step 1: CMI Filter (remove redundant features given base predictions)
            tprint_info(f"   Step 1: CMI Selection on {len(available_features)} features...")
            X_cmi, cmi_selector = cmi_feature_selection(
                X_raw, y_raw, base_predictions,
                threshold_percentile=25.0,  # Data-driven 25th percentile
                n_bins=10,
                min_samples=100
            )
            
            cmi_features = X_cmi.columns.tolist()
            tprint_success(f"   CMI kept {len(cmi_features)}/{len(available_features)} features")
            
            # Step 2: De Prado Engine (structural selection)
            if len(cmi_features) > 5:  # Only run if enough features remain
                tprint_info(f"   Step 2: De Prado Engine on {len(cmi_features)} features...")
                
                X_final, de_prado_engine = de_prado_feature_selection(
                    X_cmi, y_raw,  # Use binary target directly
                    n_estimators=300,  # Reduced for speed in Layer 4
                    max_clusters=min(8, len(cmi_features)//3),  # Fewer clusters for Layer 4
                    gain_weight=0.6,  # Slightly favor predictive power for sizing
                    depth_weight=0.4
                )
                
                final_features = X_final.columns.tolist()
                tprint_success(f"   De Prado selected {len(final_features)}/{len(cmi_features)} features")
                
                # Update available features
                available_features = final_features
                
                # Store selection info for reporting
                # Store selection info for reporting
                cmi_summary = cmi_selector.get_summary()
                
                advanced_time = time.time() - advanced_start
                tprint_info(f"   📊 Layer 4 Selection Summary:")
                tprint_info(f"      ⏱️  Total time: {advanced_time:.2f}s")
                tprint_info(f"      📉 CMI Threshold: {cmi_summary['threshold']:.6f} bits")
                tprint_info(f"      🌳 De Prado Clusters: {de_prado_engine.optimal_n_clusters_}")
                tprint_info(f"      📊 Final Feature Count: {len(final_features)}")
                tprint_info(f"      📈 Reduction: {(len(available_features)-len(final_features))/len(available_features):.1%}")
                tprint_info(f"      CMI Threshold: {cmi_summary['threshold']:.6f} bits")
                tprint_info(f"      De Prado Clusters: {de_prado_engine.optimal_n_clusters_}")
                tprint_info(f"      Final Feature Count: {len(final_features)}")
                
                # Save selection reports
                try:
                    outcomes_dir = Path("outcomes")
                    outcomes_dir.mkdir(exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    cmi_selector.get_feature_scores().to_csv(outcomes_dir / f"layer4_cmi_scores_{ts}.csv")
                    de_prado_engine.get_feature_stats().to_csv(outcomes_dir / f"layer4_deprado_stats_{ts}.csv")
                    de_prado_engine.get_report().to_csv(outcomes_dir / f"layer4_deprado_report_{ts}.csv")
                except Exception as e:
                    tprint_warning(f"   Failed to save Layer 4 selection reports: {e}")
                    
            else:
                tprint_warning(f"   Too few features after CMI ({len(cmi_features)}), skipping De Prado")
                available_features = cmi_features
                
        except Exception as e:
            tprint_warning(f"   ⚠️ Advanced feature selection failed for Layer 4: {e}")
            tprint_info(f"   Continuing with original {len(available_features)} features...")
    
    else:
        if not enable_advanced_selection:
            tprint_info("   Advanced feature selection disabled for Layer 4")
        else:
            tprint_info(f"   Skipping advanced selection for Layer 4 (only {len(available_features)} features)")
    

    tprint_info(f"📊 Using {len(available_features)} features for ExtraTrees")
    
    X = df_aligned[available_features].fillna(0)
    y = binary_target_aligned
    
    # 4. Sample weights (if enabled)
    sample_weights = None
    if use_weights:
        # Use absolute returns as weights (higher weight for larger moves)
        sample_weights = np.abs(returns_aligned)
        sample_weights = sample_weights / (sample_weights.mean() + 1e-8)  # Normalize
        tprint_info(f"📊 Using sample weights (mean: {sample_weights.mean():.3f})")
    
    # 5. Hyperparameter optimization focused on PnL and Sortino
    def objective(trial):
        # ExtraTrees hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 10, 50)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 25)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, None])
        
        # Create model
        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation with time series split
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Calculate PnL-based score instead of just AUC
        pnl_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            returns_val = returns_aligned.iloc[val_idx]
            
            # Train model
            if sample_weights is not None:
                weights_train = sample_weights.iloc[train_idx]
                model.fit(X_train, y_train, sample_weight=weights_train)
            else:
                model.fit(X_train, y_train)
            
            # Predict probabilities
            probas = model.predict_proba(X_val)[:, 1]
            
            # Calculate PnL based on predictions
            # Simple strategy: go long when prob > 0.6, short when prob < 0.4
            positions = np.where(probas > 0.6, 1, np.where(probas < 0.4, -1, 0))
            pnl = positions * returns_val
            
            # Calculate Sortino ratio (downside deviation only)
            if len(pnl) > 1:
                downside_returns = pnl[pnl < 0]
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns)
                    sortino = pnl.mean() / (downside_std + 1e-8) * np.sqrt(365 * 24 * 4)  # Annualized
                else:
                    sortino = pnl.mean() * np.sqrt(365 * 24 * 4)  # No downside risk
                
                pnl_scores.append(sortino)
            else:
                pnl_scores.append(0.0)
        
        return np.mean(pnl_scores)
    
    # Optimize hyperparameters
    n_trials = cfg.get('n_trials', 50)
    tprint_info(f"🔍 Optimizing hyperparameters with {n_trials} trials for Sortino optimization...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=HyperbandPruner()
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=300)  # 5 minute timeout
    
    best_params = study.best_params
    best_score = study.best_value
    
    tprint_success(f"✅ Best Sortino score: {best_score:.4f}")
    tprint_info(f"🎯 Best params: {best_params}")
    
    # 6. Train final model
    tprint_info("🏋️ Training final ExtraTrees model...")
    
    final_model = ExtraTreesClassifier(
        random_state=42,
        n_jobs=-1,
        **best_params
    )
    
    # Train on full dataset
    if sample_weights is not None:
        final_model.fit(X, y, sample_weight=sample_weights)
    else:
        final_model.fit(X, y)
    
    # 7. Generate predictions
    tprint_info("📊 Generating predictions...")
    
    # Class probabilities
    probas = final_model.predict_proba(X)
    
    # Predicted class
    predicted_class = final_model.predict(X)
    
    # Probability of positive return
    positive_prob = probas[:, 1]
    
    # Create output DataFrame
    predictions_df = df_aligned.copy()
    predictions_df['layer4_extratrees_prob'] = positive_prob
    predictions_df['layer4_extratrees_class'] = predicted_class
    predictions_df['layer4_extratrees_confidence'] = np.max(probas, axis=1)
    
    # Add Layer4 probability proxy for Layer5 compatibility
    predictions_df['layer4_prob'] = positive_prob
    
    # 8. Calculate performance metrics
    try:
        # Standard metrics
        auc = roc_auc_score(y, positive_prob)
        ll = log_loss(y, probas)
        brier = brier_score_loss(y, positive_prob)
        
        # PnL metrics
        positions = np.where(positive_prob > 0.6, 1, np.where(positive_prob < 0.4, -1, 0))
        pnl = positions * returns_aligned
        
        # Calculate comprehensive metrics
        total_pnl = pnl.sum()
        sharpe_ratio = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(365 * 24 * 4)
        
        # Sortino ratio
        downside_returns = pnl[pnl < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino_ratio = pnl.mean() / (downside_std + 1e-8) * np.sqrt(365 * 24 * 4)
        else:
            sortino_ratio = pnl.mean() * np.sqrt(365 * 24 * 4)
        
        # Maximum drawdown
        cumulative_pnl = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max) / (running_max + 1e-8)
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = (pnl > 0).mean()
        
        tprint_info(f"📊 Final Performance Metrics:")
        tprint_info(f"   AUC: {auc:.4f}")
        tprint_info(f"   Total PnL: {total_pnl:.4f}")
        tprint_info(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        tprint_info(f"   Sortino Ratio: {sortino_ratio:.2f}")
        tprint_info(f"   Max Drawdown: {max_drawdown:.3f}")
        tprint_info(f"   Win Rate: {win_rate:.3f}")
        
    except Exception as e:
        tprint_warning(f"⚠️ Could not calculate performance metrics: {e}")
        auc, ll, brier = 0.5, 0.693, 0.25
        total_pnl, sharpe_ratio, sortino_ratio, max_drawdown, win_rate = 0.0, 0.0, 0.0, 0.0, 0.5
    
    # 9. Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(15)
    tprint_info(f"🏆 Top 15 features by importance:")
    for _, row in top_features.iterrows():
        tprint_info(f"   {row['feature']}: {row['importance']:.4f}")
    
    # 10. Save model and metadata
    metadata = {
        'model_type': 'ExtraTreesClassifier',
        'target_col': target_col,
        'use_raw_returns': use_raw_returns,
        'use_weights': use_weights,
        'n_features': len(available_features),
        'feature_names': available_features,
        'best_params': best_params,
        'best_sortino_score': best_score,
        'final_auc': auc,
        'final_log_loss': ll,
        'final_brier_score': brier,
        'total_pnl': total_pnl,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'feature_importance': feature_importance.to_dict(),
        'training_samples': len(X),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to outcomes
    try:
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metadata
        with open(outcomes_dir / f"layer4_extratrees_metadata_{ts}.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save feature importance
        feature_importance.to_csv(outcomes_dir / f"layer4_extratrees_feature_importance_{ts}.csv", index=False)
        
        # Save predictions
        predictions_cols = ['layer4_extratrees_prob', 'layer4_extratrees_class', 'layer4_extratrees_confidence', 'layer4_prob']
        predictions_df[predictions_cols].to_csv(outcomes_dir / f"layer4_extratrees_predictions_{ts}.csv")
        
        tprint_success(f"💾 Layer4 ExtraTrees results saved with timestamp {ts}")
        
    except Exception as e:
        tprint_error(f"❌ Failed to save results: {e}")
    
    return predictions_df, metadata


# Legacy compatibility function
def train_layer4_oof(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    l3_prob_col: str = 'meta_prob',
    target_col: str = 'realized_return',
    return_col: str = 'realized_return',
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None,
    # Deprecated parameters - kept for backward compatibility only
    l3_models_metadata: Optional[Dict] = None,
    l3_quantile_thresholds: Optional[List[float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Legacy compatibility wrapper for Layer4 ExtraTrees training.
    
    This function maintains the same interface as the old Layer4
    but now uses ExtraTrees optimized for PnL and Sortino.
    """
    
    tprint_info("🔄 Using ExtraTrees model for Layer4 (PnL & Sortino optimized)")
    
    # Extract configuration
    cfg = config or {}
    use_raw_returns = cfg.get('use_raw_returns', True)
    use_weights = cfg.get('use_weights', True)
    
    # Call the new training function
    predictions_df, metadata = train_layer4_extratrees(
        df=market_data,
        layer3_predictions=oof_df,
        target_col=target_col,
        prob_col=l3_prob_col,
        use_raw_returns=use_raw_returns,
        use_weights=use_weights,
        n_folds=n_folds,
        config=config
    )
    
    # Add legacy column names for compatibility
    if 'layer4_extratrees_prob' in predictions_df.columns:
        predictions_df['layer4_weight'] = predictions_df['layer4_extratrees_prob']
    
    if 'layer4_extratrees_confidence' in predictions_df.columns:
        predictions_df['layer4_return'] = predictions_df['layer4_extratrees_confidence']  # For compatibility
    
    return predictions_df, metadata
