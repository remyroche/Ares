"""
Layer 4 Unified Feature Generator

Consolidates all Layer 4 feature generation into a single, configurable system
for position sizing and risk management.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path
from datetime import datetime
import json
import time

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from scipy.stats import spearmanr, norm
import statsmodels.api as sm

# Import feature registry
from .feature_registry import get_layer4_feature_patterns_by_category, get_core_layer4_features

# Import external feature generators
try:
    from src.feature_generation.categories.ensemble_disagreement import EnsembleDisagreementFeatures
    ENSEMBLE_DISAGREEMENT_AVAILABLE = True
except ImportError:
    ENSEMBLE_DISAGREEMENT_AVAILABLE = False
    tprint_warning("⚠️ Ensemble disagreement features not available")

try:
    from src.training.steps.labeling.contextual_residual_features import generate_contextual_residual_features
    CONTEXTUAL_FEATURES_AVAILABLE = True
except ImportError:
    CONTEXTUAL_FEATURES_AVAILABLE = False
    tprint_warning("⚠️ Contextual residual features not available")

try:
    from src.training.steps.labeling.de_prado_feature_engine import DePradoFeatureEngine
    DE_PRADO_AVAILABLE = True
except ImportError:
    DE_PRADO_AVAILABLE = False
    tprint_warning("⚠️ De Prado feature engine not available")


class Layer4FeatureGenerator:
    """
    Unified Layer 4 Feature Generator for Position Sizing & Risk Management.
    
    Consolidates all feature generation logic including:
    - Performance features (PSR, precision, entropy)
    - Regime features (volatility, trend, market state)
    - Market features (relative strength, VWAP, drawdown)
    - Model features (disagreement, probability products)
    - Structural features (break scores, change points)
    - Time features (temporal patterns)
    - Contextual features (residuals, harmonization)
    """
    
    def __init__(
        self,
        window: int = 50,
        span: int = 20,
        prior_sr: float = 0.0,
        prior_weight: int = 10,
        min_psr_obs: int = 20,
        config: Optional[Dict[str, Any]] = None
    ):
        self.window = window
        self.span = span
        self.prior_sr = prior_sr
        self.prior_weight = prior_weight
        self.min_psr_obs = min_psr_obs
        self.config = config or {}
        
        # Feature category enable flags
        self.enable_performance = self.config.get('enable_performance', True)
        self.enable_regime = self.config.get('enable_regime', True)
        self.enable_market = self.config.get('enable_market', True)
        self.enable_technical = self.config.get('enable_technical', True)
        self.enable_structural = self.config.get('enable_structural', True)
        self.enable_model = self.config.get('enable_model', True)
        self.enable_time = self.config.get('enable_time', True)
        self.enable_contextual = self.config.get('enable_contextual', True)
        
        # Initialize sub-generators
        self._init_sub_generators()
    
    def _init_sub_generators(self):
        """Initialize sub-generators for different feature types."""
        # Ensemble disagreement generator
        if ENSEMBLE_DISAGREEMENT_AVAILABLE and self.enable_model:
            self.disagreement_calculator = EnsembleDisagreementFeatures()
        else:
            self.disagreement_calculator = None
            
        # De Prado feature engine
        if DE_PRADO_AVAILABLE and self.enable_contextual:
            self.de_prado_engine = DePradoFeatureEngine()
        else:
            self.de_prado_engine = None
    
    # ------------------------------------------------------------------
    # Performance Features
    # ------------------------------------------------------------------
    def bayesian_psr(self, returns: np.ndarray, benchmark_sr: float = 0.0) -> float:
        """Bayesian-shrunk PSR (De Prado)."""
        r = np.asarray(returns)
        n = len(r)
    
        if n < self.min_psr_obs:
            return 0.0
    
        mean = r.mean()
        std = r.std(ddof=1) + 1e-9
        sample_sr = mean / std
    
        # Bayesian shrinkage toward long-term SR prior
        shrunk_sr = (
            sample_sr * n + self.prior_sr * self.prior_weight
        ) / (n + self.prior_weight)
    
        skew = pd.Series(r).skew()
        kurt = pd.Series(r).kurtosis()
    
        # Variance of Sharpe estimator (AFML, corrected)
        var_sr = (
            1
            - skew * shrunk_sr
            + ((kurt - 1.0) / 4.0) * shrunk_sr ** 2
        ) / (n - 1)
    
        if var_sr <= 0 or not np.isfinite(var_sr):
            return 0.0
    
        sigma_sr = np.sqrt(var_sr)
        return norm.cdf((shrunk_sr - benchmark_sr) / sigma_sr)
    
    def get_sadf_proxy(self, price: pd.Series) -> pd.Series:
        """Rolling ADF-style explosiveness proxy."""
        log_p = np.log(price.replace(0, np.nan)).dropna()
    
        def adf_tstat(x):
            if len(x) < 20:
                return 0.0
    
            y = x.values
            dy = np.diff(y)
            y_lag = y[:-1]
    
            try:
                res = sm.OLS(dy, sm.add_constant(y_lag)).fit(disp=False)
                return res.tvalues[1]
            except Exception:
                return 0.0
    
        return (
            log_p
            .rolling(self.window)
            .apply(adf_tstat, raw=False)
            .reindex(price.index)
            .fillna(0.0)
        )
    
    @staticmethod
    def binary_entropy(errors: np.ndarray) -> float:
        """Binary entropy for model stability."""
        p = errors.mean()
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    
    def calculate_past_precision(self, df: pd.DataFrame, target_col: str, prob_col: str, window: int = 100) -> pd.Series:
        """Calculate past precision (rolling accuracy)."""
        if target_col not in df.columns or prob_col not in df.columns:
            return pd.Series(0.5, index=df.index)
        
        # Binary target from returns
        returns = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
        binary_target = (returns > 0).astype(int)
        
        # Binary predictions from probabilities
        binary_pred = (df[prob_col] > 0.5).astype(int)
        
        # Rolling accuracy
        correct = (binary_target == binary_pred).astype(int)
        precision = correct.rolling(window=window, min_periods=10).mean()
        
        return precision.fillna(0.5)
    
    # ------------------------------------------------------------------
    # Structural Break Features
    # ------------------------------------------------------------------
    def calculate_structural_break_scores(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate structural break scores using SADF and CUSUM filters."""
        try:
            prices = df[price_col].values
            returns = np.diff(np.log(prices + 1e-8))
            
            # SADF scores
            sadf_scores = []
            window_size = min(100, len(returns) // 4)
            
            for i in range(window_size, len(returns)):
                window_returns = returns[i-window_size:i]
                
                if len(window_returns) > 10:
                    x = window_returns[:-1]
                    y = window_returns[1:]
                    
                    if len(x) > 1 and np.std(x) > 1e-8:
                        beta = np.cov(x, y)[0, 1] / np.var(x)
                        alpha = np.mean(y) - beta * np.mean(x)
                        residuals = y - (alpha + beta * x)
                        
                        if np.std(residuals) > 1e-8:
                            t_stat = beta / (np.std(residuals) / np.sqrt(len(x) * np.var(x)))
                            sadf_scores.append(abs(t_stat))
                        else:
                            sadf_scores.append(0.0)
                    else:
                        sadf_scores.append(0.0)
                else:
                    sadf_scores.append(0.0)
            
            # CUSUM scores
            cusum_scores = []
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return > 1e-8:
                cusum_pos = 0
                cusum_neg = 0
                
                for ret in returns:
                    cusum_pos = max(0, cusum_pos + (ret - mean_return))
                    cusum_neg = min(0, cusum_neg + (ret - mean_return))
                    cusum_scores.append(max(abs(cusum_pos), abs(cusum_neg)) / std_return)
            else:
                cusum_scores = [0.0] * len(returns)
            
            # Normalize and align
            sadf_norm = np.array(sadf_scores + [0.0] * (len(df) - len(sadf_scores)))
            sadf_norm = sadf_norm / (np.max(sadf_norm) + 1e-8)
            
            cusum_norm = np.array(cusum_scores + [0.0] * (len(df) - len(cusum_scores)))
            cusum_norm = cusum_norm / (np.max(cusum_norm) + 1e-8)
            
            result_df = pd.DataFrame(index=df.index)
            result_df['sadf_score_norm'] = sadf_norm[:len(df)]
            result_df['cusum_score_norm'] = cusum_norm[:len(df)]
            
            return result_df
            
        except Exception as e:
            tprint_error(f"Error calculating structural break scores: {e}")
            result_df = pd.DataFrame(index=df.index)
            result_df['sadf_score_norm'] = 0.0
            result_df['cusum_score_norm'] = 0.0
            return result_df
    
    # ------------------------------------------------------------------
    # Market Features
    # ------------------------------------------------------------------
    def calculate_relative_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate relative strength features."""
        result_df = pd.DataFrame(index=df.index)
        
        if 'close' in df.columns:
            close = df['close']
            
            # Simple moving averages for relative strength
            ma_short = close.rolling(20).mean()
            ma_long = close.rolling(50).mean()
            
            result_df['relative_strength_ma'] = (close - ma_long) / (ma_long + 1e-8)
            result_df['relative_strength_short'] = (close - ma_short) / (ma_short + 1e-8)
        
        return result_df.fillna(0.0)
    
    # ------------------------------------------------------------------
    # Main Feature Generation Method
    # ------------------------------------------------------------------
    def generate_all_features(
        self,
        df: pd.DataFrame,
        layer3_predictions: Optional[pd.DataFrame] = None,
        target_col: str = 'realized_return',
        prob_col: str = 'meta_prob',
        raw_price_col: str = 'close',
        denoised_price_col: str = 'denoised_price',
        use_raw_returns: bool = True
    ) -> pd.DataFrame:
        """
        Generate all Layer 4 features in a unified manner.
        
        Args:
            df: Market data DataFrame
            layer3_predictions: Layer3 OOF predictions DataFrame
            target_col: Target column name
            prob_col: Layer3 probability column name
            raw_price_col: Raw price column name
            denoised_price_col: Denoised price column name
            use_raw_returns: Whether to use raw returns
            
        Returns:
            DataFrame with all Layer 4 features
        """
        start_time = time.time()
        tprint_info("🔧 Generating unified Layer 4 features...")
        
        # Start with base dataframe
        features_df = df.copy()
        
        # Combine with Layer 3 predictions if provided
        if layer3_predictions is not None:
            features_df = features_df.join(layer3_predictions, how='inner', rsuffix='_l3')
        
        # Generate features by category
        feature_counts = {}
        
        # 1. Performance Features
        if self.enable_performance:
            perf_features = self._generate_performance_features(features_df, target_col, prob_col)
            features_df = pd.concat([features_df, perf_features], axis=1)
            feature_counts['performance'] = len(perf_features.columns)
        
        # 2. Regime Features
        if self.enable_regime:
            regime_features = self._generate_regime_features(features_df, raw_price_col)
            features_df = pd.concat([features_df, regime_features], axis=1)
            feature_counts['regime'] = len(regime_features.columns)
        
        # 3. Market Features
        if self.enable_market:
            market_features = self._generate_market_features(
                features_df, raw_price_col, denoised_price_col
            )
            features_df = pd.concat([features_df, market_features], axis=1)
            feature_counts['market'] = len(market_features.columns)
        
        # 4. Technical Features
        if self.enable_technical:
            tech_features = self._generate_technical_features(features_df)
            features_df = pd.concat([features_df, tech_features], axis=1)
            feature_counts['technical'] = len(tech_features.columns)
        
        # 5. Structural Features
        if self.enable_structural:
            struct_features = self._generate_structural_features(features_df)
            features_df = pd.concat([features_df, struct_features], axis=1)
            feature_counts['structural'] = len(struct_features.columns)
        
        # 6. Model Features (Layer 3 inputs and disagreement)
        if self.enable_model and layer3_predictions is not None:
            model_features = self._generate_model_features(
                features_df, layer3_predictions, prob_col
            )
            features_df = pd.concat([features_df, model_features], axis=1)
            feature_counts['model'] = len(model_features.columns)
        
        # 7. Time Features
        if self.enable_time:
            time_features = self._generate_time_features(features_df)
            features_df = pd.concat([features_df, time_features], axis=1)
            feature_counts['time'] = len(time_features.columns)
        
        # 8. Contextual Features (advanced)
        if self.enable_contextual and layer3_predictions is not None:
            context_features = self._generate_contextual_features(
                features_df, layer3_predictions, prob_col, target_col
            )
            if context_features is not None and hasattr(context_features, 'columns') and len(context_features.columns) > 0:
                features_df = pd.concat([features_df, context_features], axis=1)
                feature_counts['contextual'] = len(context_features.columns)
        
        # Clean up infinite and NaN values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
            features_df[col] = features_df[col].fillna(features_df[col].rolling(20, min_periods=1).mean())
            features_df[col] = features_df[col].fillna(0.0)
        
        # Log completion
        total_time = time.time() - start_time
        total_features = sum(feature_counts.values())
        
        tprint_success(f"✅ Generated {total_features} Layer 4 features in {total_time:.2f}s")
        for category, count in feature_counts.items():
            if count > 0:
                tprint_info(f"   {category.capitalize()}: {count} features")
        
        return features_df
    
    def _generate_performance_features(self, df: pd.DataFrame, target_col: str, prob_col: str) -> pd.DataFrame:
        """Generate performance-related features."""
        features = pd.DataFrame(index=df.index)
        
        # Bayesian PSR
        if 'primary_ret' in df.columns:
            features["perf_bayesian_psr"] = (
                df["primary_ret"]
                .rolling(self.window)
                .apply(self.bayesian_psr, raw=True)
            )
            
            # PSR momentum
            features["perf_psr_trend"] = (
                features["perf_bayesian_psr"]
                .diff()
                .ewm(span=10, adjust=False)
                .mean()
            )
        
        # Past precision
        features["past_precision"] = self.calculate_past_precision(df, target_col, prob_col)
        
        # Average probability product (if multiple Layer 3 models)
        layer3_prob_cols = [c for c in df.columns if c.startswith('meta_prob_') or c == prob_col]
        if len(layer3_prob_cols) >= 2:
            prob_matrix = df[layer3_prob_cols].values
            n_models = len(layer3_prob_cols)
            pairwise_products = []
            for i in range(n_models):
                for j in range(i+1, n_models):
                    pairwise_products.append(prob_matrix[:, i] * prob_matrix[:, j])
            features['avg_prob_product'] = np.mean(pairwise_products, axis=0)
        elif prob_col in df.columns:
            features['avg_prob_product'] = df[prob_col]
        
        # Model entropy (if predictions available)
        if 'y_true' in df.columns and 'y_pred' in df.columns:
            errors = (df["y_true"] != df["y_pred"]).astype(int)
            features["perf_entropy"] = (
                errors
                .rolling(self.window)
                .apply(self.binary_entropy, raw=True)
            )
        
        return features.fillna(0.0)
    
    def _generate_regime_features(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Generate regime-related features."""
        features = pd.DataFrame(index=df.index)
        
        if price_col not in df.columns:
            return features
        
        close = df[price_col]
        log_ret = np.log(close / close.shift(1))
        
        # Volatility features
        rv_short = log_ret.rolling(window=12).std() * np.sqrt(12)
        rv_long = log_ret.rolling(window=200).std()
        
        features['vol_long'] = rv_long
        features['vol_ratio'] = rv_short / (rv_long + 1e-8)
        
        # SADF proxy
        features["regime_sadf"] = self.get_sadf_proxy(close)
        
        return features.fillna(0.0)
    
    def _generate_market_features(
        self, df: pd.DataFrame, raw_price_col: str, denoised_price_col: str
    ) -> pd.DataFrame:
        """Generate market-related features."""
        features = pd.DataFrame(index=df.index)
        
        # Market stretch (raw vs denoised)
        if raw_price_col in df.columns and denoised_price_col in df.columns:
            stretch = np.log(
                (df[raw_price_col] + 1e-9) /
                (df[denoised_price_col] + 1e-9)
            )
            features["market_stretch"] = stretch.clip(-5, 5)
            
            # Noise persistence
            features["noise_persistence"] = (
                (df[raw_price_col] - df[denoised_price_col])
                .rolling(self.span)
                .std()
            )
        
        # Relative strength
        rel_strength = self.calculate_relative_strength(df)
        for col in rel_strength.columns:
            features[col] = rel_strength[col]
        
        return features.fillna(0.0)
    
    def _generate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical analysis features."""
        features = pd.DataFrame(index=df.index)
        
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            return features
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # ADX proxy
        up_move = high.diff()
        down_move = low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr = (high - low) / close
        tr_smooth = tr.rolling(window=14).sum()
        plus_di = pd.Series(plus_dm, index=df.index).rolling(window=14).sum() / (tr_smooth + 1e-8)
        minus_di = pd.Series(minus_dm, index=df.index).rolling(window=14).sum() / (tr_smooth + 1e-8)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        features['adx_proxy'] = dx.rolling(window=14).mean()
        
        # Choppiness index
        chop_window = 20
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr_series = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        sum_tr = tr_series.rolling(chop_window).sum()
        max_hi = high.rolling(chop_window).max()
        min_lo = low.rolling(chop_window).min()
        range_hl = max_hi - min_lo
        
        features['choppiness_index'] = 100 * np.log10(sum_tr / (range_hl + 1e-8)) / np.log10(chop_window)
        
        # Variance ratio
        log_ret = np.log(close / close.shift(1))
        vr_window = 50
        r_20 = log_ret.rolling(20).sum()
        r_10 = log_ret.rolling(10).sum()
        var_20 = r_20.rolling(vr_window).var()
        var_10 = r_10.rolling(vr_window).var()
        features['variance_ratio'] = var_20 / (2 * var_10 + 1e-8)
        
        # Efficiency ratio
        er_window = 10
        change = (close - close.shift(er_window)).abs()
        volatility = close.diff().abs().rolling(er_window).sum()
        features['efficiency_ratio'] = change / (volatility + 1e-8)
        
        return features.fillna(0.0)
    
    def _generate_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate structural break and drawdown features."""
        features = pd.DataFrame(index=df.index)
        
        # Structural break scores
        struct_scores = self.calculate_structural_break_scores(df)
        for col in struct_scores.columns:
            features[col] = struct_scores[col]
        
        return features.fillna(0.0)
    
    def _generate_model_features(
        self, df: pd.DataFrame, layer3_predictions: pd.DataFrame, prob_col: str
    ) -> pd.DataFrame:
        """Generate model-related features including disagreement."""
        features = pd.DataFrame(index=df.index)
        
        # Disagreement features
        if self.disagreement_calculator is not None:
            model_cols = [c for c in df.columns if c.startswith('model_')]
            if len(model_cols) > 1:
                model_predictions = {col: df[col].values for col in model_cols}
                model_probabilities = {col: df[col].values for col in model_cols}
                
                disagreement_df = self.disagreement_calculator.calculate_disagreement_features(
                    model_predictions, model_probabilities
                )
                
                # Handle both DataFrame and dict returns
                if isinstance(disagreement_df, pd.DataFrame):
                    for col in disagreement_df.columns:
                        features[col] = disagreement_df[col]
                elif isinstance(disagreement_df, dict):
                    for col, values in disagreement_df.items():
                        features[col] = values
        
        return features.fillna(0.0)
    
    def _generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based features."""
        features = pd.DataFrame(index=df.index)
        
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour
            day_of_week = df.index.dayofweek
            
            features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            features['is_weekend'] = (day_of_week >= 5).astype(float)
            
            # Session features
            features['is_session_start'] = ((hour >= 8) & (hour <= 10)).astype(float)
            features['is_session_end'] = ((hour >= 16) & (hour <= 18)).astype(float)
        
        return features.fillna(0.0)
    
    def _generate_contextual_features(
        self, df: pd.DataFrame, layer3_predictions: pd.DataFrame, prob_col: str, target_col: str
    ) -> Optional[pd.DataFrame]:
        """Generate contextual residual features."""
        if not CONTEXTUAL_FEATURES_AVAILABLE:
            return None
        
        try:
            layer3_prob_cols = [c for c in df.columns if c.startswith('meta_prob_') or c == prob_col]
            if len(layer3_prob_cols) <= 3:
                return None
            
            predictions_df = df[layer3_prob_cols + [prob_col]].copy()
            predictions_df = predictions_df.rename(columns={prob_col: "target"})
            
            residual_features, _ = generate_contextual_residual_features(
                predictions_df=predictions_df,
                target_col="target",
                harmonization_type=self.config.get("harmonization_type", "direction"),
                max_residual_features=self.config.get("max_residual_features", 50)
            )
            
            return residual_features
            
        except Exception as e:
            tprint_warning(f"⚠️ Contextual feature generation failed: {e}")
            return None
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of available features by category."""
        from .feature_registry import validate_layer4_features
        
        validation_result = validate_layer4_features(df)
        
        # Count features by category
        category_patterns = get_layer4_feature_patterns_by_category()
        category_counts = {}
        
        for category, patterns in category_patterns.items():
            count = 0
            for pattern in patterns:
                matching_cols = [c for c in df.columns if pattern in c]
                count += len(matching_cols)
            category_counts[category] = count
        
        return {
            'validation_result': validation_result,
            'category_counts': category_counts,
            'total_features': len(df.columns),
            'enabled_categories': [cat for cat, enabled in [
                ('performance', self.enable_performance),
                ('regime', self.enable_regime),
                ('market', self.enable_market),
                ('technical', self.enable_technical),
                ('structural', self.enable_structural),
                ('model', self.enable_model),
                ('time', self.enable_time),
                ('contextual', self.enable_contextual)
            ] if enabled]
        }
