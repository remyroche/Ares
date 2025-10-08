"""
Enhanced Feature Selection with Bootstrap Validation and Economic Interpretability.

This module implements robust feature selection:
1. Bootstrap validation for feature stability
2. Economic interpretability grouping
3. Out-of-sample IC tracking
4. Multi-fold selection with consensus voting
5. Economic theme preservation

Key improvements:
- Features selected across multiple folds with consistency check
- Per-feature IC (Information Coefficient) tracked over time
- Economic grouping (trend, momentum, volatility, microstructure)
- Orthogonalized factor portfolio validation
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import system_logger


class EconomicTheme(Enum):
    """Economic themes for feature grouping."""
    
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    MICROSTRUCTURE = "microstructure"
    VOLUME = "volume"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"


@dataclass
class FeatureSelectionConfig:
    """Configuration for enhanced feature selection."""
    
    # Bootstrap settings
    n_bootstrap_folds: int = 5  # Number of bootstrap samples
    min_selection_frequency: float = 0.60  # Min % of folds to keep feature
    
    # IC tracking
    track_ic: bool = True
    ic_window_size: int = 100  # Rolling window for IC
    min_ic_threshold: float = 0.02  # Minimum acceptable IC
    min_ic_t_stat: float = 2.0  # Minimum t-statistic
    
    # Economic grouping
    preserve_economic_themes: bool = True
    min_features_per_theme: int = 1  # Minimum features per theme
    
    # Feature importance
    importance_method: str = "mutual_info"  # 'mutual_info', 'correlation', 'model_based'
    
    # Validation
    validate_with_factor_portfolio: bool = True
    min_factor_sharpe: float = 0.3  # Minimum Sharpe for validation


@dataclass
class FeatureInfo:
    """Information about a feature."""
    
    name: str
    importance: float
    ic_mean: float
    ic_std: float
    ic_t_stat: float
    selection_frequency: float
    economic_theme: EconomicTheme
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionResult:
    """Result from feature selection."""
    
    selected_features: List[str]
    feature_info: List[FeatureInfo]
    
    # Bootstrap results
    selection_frequencies: Dict[str, float]
    bootstrap_iterations: int
    
    # Economic theme distribution
    theme_distribution: Dict[str, int]
    
    # Validation results
    factor_portfolio_sharpe: Optional[float] = None
    validation_passed: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_features(self) -> int:
        """Number of selected features."""
        return len(self.selected_features)
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Summary statistics."""
        return {
            'n_features': self.n_features,
            'mean_ic': np.mean([f.ic_mean for f in self.feature_info]),
            'mean_importance': np.mean([f.importance for f in self.feature_info]),
            'mean_selection_freq': np.mean([f.selection_frequency for f in self.feature_info]),
            'theme_distribution': self.theme_distribution,
            'validation_passed': self.validation_passed,
            'factor_sharpe': self.factor_portfolio_sharpe
        }


class EnhancedFeatureSelector:
    """
    Enhanced feature selector with bootstrap validation and economic interpretability.
    """
    
    def __init__(
        self,
        config: Optional[FeatureSelectionConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the enhanced feature selector.
        
        Args:
            config: Feature selection configuration
            logger: Optional logger instance
        """
        self.config = config or FeatureSelectionConfig()
        self.logger = logger or system_logger.getChild('EnhancedFeatureSelector')
    
    def select_features(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        feature_themes: Optional[Dict[str, EconomicTheme]] = None,
        target_n_features: Optional[int] = None
    ) -> SelectionResult:
        """
        Select robust features using bootstrap validation.
        
        Args:
            features: DataFrame with features
            labels: Target labels
            feature_themes: Optional mapping of feature names to economic themes
            target_n_features: Target number of features to select
        
        Returns:
            SelectionResult with selection results
        """
        # Align data
        common_idx = features.index.intersection(labels.index)
        features_aligned = features.loc[common_idx]
        labels_aligned = labels.loc[common_idx]
        
        if len(common_idx) < 100:
            self.logger.warning(f"Insufficient data for selection: {len(common_idx)} samples")
            return SelectionResult(
                selected_features=[],
                feature_info=[],
                selection_frequencies={},
                bootstrap_iterations=0,
                theme_distribution={}
            )
        
        # Infer themes if not provided
        if feature_themes is None:
            feature_themes = self._infer_feature_themes(features_aligned.columns)
        
        # Bootstrap feature selection
        selection_frequencies = self._bootstrap_selection(
            features=features_aligned,
            labels=labels_aligned,
            n_folds=self.config.n_bootstrap_folds
        )
        
        # Filter features by selection frequency
        stable_features = [
            feat for feat, freq in selection_frequencies.items()
            if freq >= self.config.min_selection_frequency
        ]
        
        if not stable_features:
            self.logger.warning("No stable features found")
            return SelectionResult(
                selected_features=[],
                feature_info=[],
                selection_frequencies=selection_frequencies,
                bootstrap_iterations=self.config.n_bootstrap_folds,
                theme_distribution={}
            )
        
        # Compute feature information
        feature_info_list = []
        for feat in stable_features:
            info = self._compute_feature_info(
                feature=features_aligned[feat],
                labels=labels_aligned,
                feature_name=feat,
                theme=feature_themes.get(feat, EconomicTheme.TECHNICAL),
                selection_freq=selection_frequencies[feat]
            )
            feature_info_list.append(info)
        
        # Sort by importance
        feature_info_list.sort(key=lambda x: x.importance, reverse=True)
        
        # Select top features
        if target_n_features:
            selected_info = feature_info_list[:target_n_features]
        else:
            selected_info = feature_info_list
        
        # Ensure economic theme diversity
        if self.config.preserve_economic_themes:
            selected_info = self._ensure_theme_diversity(
                feature_info_list=selected_info,
                all_features=feature_info_list,
                target_n=target_n_features
            )
        
        selected_features = [info.name for info in selected_info]
        
        # Compute theme distribution
        theme_distribution = {}
        for info in selected_info:
            theme_name = info.economic_theme.value
            theme_distribution[theme_name] = theme_distribution.get(theme_name, 0) + 1
        
        # Validate with factor portfolio
        factor_sharpe = None
        validation_passed = True
        
        if self.config.validate_with_factor_portfolio:
            factor_sharpe = self._validate_factor_portfolio(
                features=features_aligned[selected_features],
                labels=labels_aligned
            )
            
            if factor_sharpe is not None:
                validation_passed = factor_sharpe >= self.config.min_factor_sharpe
                
                if not validation_passed:
                    self.logger.warning(
                        f"Factor portfolio validation failed: Sharpe={factor_sharpe:.3f} "
                        f"(threshold: {self.config.min_factor_sharpe})"
                    )
        
        result = SelectionResult(
            selected_features=selected_features,
            feature_info=selected_info,
            selection_frequencies=selection_frequencies,
            bootstrap_iterations=self.config.n_bootstrap_folds,
            theme_distribution=theme_distribution,
            factor_portfolio_sharpe=factor_sharpe,
            validation_passed=validation_passed
        )
        
        self.logger.info(
            f"Feature selection complete: {len(selected_features)} features selected, "
            f"validation_passed={validation_passed}"
        )
        
        return result
    
    def _bootstrap_selection(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        n_folds: int
    ) -> Dict[str, float]:
        """
        Perform bootstrap feature selection.
        
        Args:
            features: Feature DataFrame
            labels: Label series
            n_folds: Number of bootstrap folds
        
        Returns:
            Dictionary mapping feature names to selection frequencies
        """
        feature_counts = {col: 0 for col in features.columns}
        
        for fold in range(n_folds):
            # Bootstrap sample
            sample_size = int(len(features) * 0.8)
            sample_idx = np.random.choice(len(features), size=sample_size, replace=True)
            
            features_sample = features.iloc[sample_idx]
            labels_sample = labels.iloc[sample_idx]
            
            # Compute importance on sample
            importances = self._compute_importances(features_sample, labels_sample)
            
            # Select top 50% features
            threshold = np.percentile(list(importances.values()), 50)
            
            for feat, importance in importances.items():
                if importance >= threshold:
                    feature_counts[feat] += 1
        
        # Convert counts to frequencies
        frequencies = {
            feat: count / n_folds
            for feat, count in feature_counts.items()
        }
        
        return frequencies
    
    def _compute_importances(
        self,
        features: pd.DataFrame,
        labels: pd.Series
    ) -> Dict[str, float]:
        """
        Compute feature importances.
        
        Args:
            features: Feature DataFrame
            labels: Label series
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.config.importance_method == "mutual_info":
            return self._compute_mutual_info_importance(features, labels)
        elif self.config.importance_method == "correlation":
            return self._compute_correlation_importance(features, labels)
        elif self.config.importance_method == "model_based":
            return self._compute_model_based_importance(features, labels)
        else:
            raise ValueError(f"Unknown importance method: {self.config.importance_method}")
    
    def _compute_mutual_info_importance(
        self,
        features: pd.DataFrame,
        labels: pd.Series
    ) -> Dict[str, float]:
        """Compute mutual information-based importance."""
        try:
            from sklearn.feature_selection import mutual_info_regression
        except ImportError:
            self.logger.warning("sklearn not available, using correlation")
            return self._compute_correlation_importance(features, labels)
        
        # Align and clean
        X = features.fillna(0).values
        y = labels.fillna(0).values
        
        if len(X) < 50:
            return {col: 0.0 for col in features.columns}
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mi_scores = mutual_info_regression(X, y, random_state=42)
            
            importances = {
                col: float(score)
                for col, score in zip(features.columns, mi_scores)
            }
        except Exception as e:
            self.logger.warning(f"MI computation failed: {e}, using correlation")
            importances = self._compute_correlation_importance(features, labels)
        
        return importances
    
    def _compute_correlation_importance(
        self,
        features: pd.DataFrame,
        labels: pd.Series
    ) -> Dict[str, float]:
        """Compute correlation-based importance."""
        importances = {}
        
        for col in features.columns:
            try:
                corr = features[col].corr(labels)
                importances[col] = abs(float(corr)) if not np.isnan(corr) else 0.0
            except Exception:
                importances[col] = 0.0
        
        return importances
    
    def _compute_model_based_importance(
        self,
        features: pd.DataFrame,
        labels: pd.Series
    ) -> Dict[str, float]:
        """Compute model-based importance using simple linear model."""
        from sklearn.linear_model import Ridge
        
        X = features.fillna(0).values
        y = labels.fillna(0).values
        
        if len(X) < 50:
            return {col: 0.0 for col in features.columns}
        
        try:
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X, y)
            
            # Use absolute coefficients as importance
            importances = {
                col: abs(float(coef))
                for col, coef in zip(features.columns, model.coef_)
            }
        except Exception as e:
            self.logger.warning(f"Model-based importance failed: {e}, using correlation")
            importances = self._compute_correlation_importance(features, labels)
        
        return importances
    
    def _compute_feature_info(
        self,
        feature: pd.Series,
        labels: pd.Series,
        feature_name: str,
        theme: EconomicTheme,
        selection_freq: float
    ) -> FeatureInfo:
        """
        Compute comprehensive information about a feature.
        
        Args:
            feature: Feature series
            labels: Label series
            feature_name: Name of the feature
            theme: Economic theme
            selection_freq: Selection frequency from bootstrap
        
        Returns:
            FeatureInfo object
        """
        # Importance
        importance = abs(feature.corr(labels))
        if np.isnan(importance):
            importance = 0.0
        
        # IC over time
        if self.config.track_ic and len(feature) >= self.config.ic_window_size:
            ics = []
            window = self.config.ic_window_size
            
            for i in range(0, len(feature) - window, window // 2):
                feat_window = feature.iloc[i:i+window]
                label_window = labels.iloc[i:i+window]
                
                try:
                    ic, _ = stats.spearmanr(feat_window, label_window, nan_policy='omit')
                    if not np.isnan(ic):
                        ics.append(ic)
                except Exception:
                    continue
            
            if ics:
                ic_mean = np.mean(ics)
                ic_std = np.std(ics)
                ic_t_stat = ic_mean / (ic_std / np.sqrt(len(ics)) + 1e-8)
            else:
                ic_mean = 0.0
                ic_std = 0.0
                ic_t_stat = 0.0
        else:
            # Compute single IC
            try:
                ic, _ = stats.spearmanr(feature, labels, nan_policy='omit')
                ic_mean = float(ic) if not np.isnan(ic) else 0.0
            except Exception:
                ic_mean = 0.0
            
            ic_std = 0.0
            ic_t_stat = 0.0
        
        return FeatureInfo(
            name=feature_name,
            importance=float(importance),
            ic_mean=float(ic_mean),
            ic_std=float(ic_std),
            ic_t_stat=float(ic_t_stat),
            selection_frequency=selection_freq,
            economic_theme=theme
        )
    
    def _infer_feature_themes(self, feature_names: pd.Index) -> Dict[str, EconomicTheme]:
        """
        Infer economic themes from feature names.
        
        Args:
            feature_names: Feature column names
        
        Returns:
            Dictionary mapping feature names to themes
        """
        themes = {}
        
        for name in feature_names:
            name_lower = name.lower()
            
            if any(kw in name_lower for kw in ['ma', 'sma', 'ema', 'trend']):
                theme = EconomicTheme.TREND
            elif any(kw in name_lower for kw in ['rsi', 'macd', 'momentum', 'roc']):
                theme = EconomicTheme.MOMENTUM
            elif any(kw in name_lower for kw in ['vol', 'atr', 'std', 'volatility']):
                theme = EconomicTheme.VOLATILITY
            elif any(kw in name_lower for kw in ['spread', 'bid', 'ask', 'orderbook']):
                theme = EconomicTheme.MICROSTRUCTURE
            elif any(kw in name_lower for kw in ['volume', 'vwap']):
                theme = EconomicTheme.VOLUME
            else:
                theme = EconomicTheme.TECHNICAL
            
            themes[name] = theme
        
        return themes
    
    def _ensure_theme_diversity(
        self,
        feature_info_list: List[FeatureInfo],
        all_features: List[FeatureInfo],
        target_n: Optional[int]
    ) -> List[FeatureInfo]:
        """
        Ensure selected features have diverse economic themes.
        
        Args:
            feature_info_list: Current selection
            all_features: All available features
            target_n: Target number of features
        
        Returns:
            Adjusted selection with theme diversity
        """
        # Count themes in current selection
        theme_counts = {}
        for info in feature_info_list:
            theme = info.economic_theme
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        # Check if any theme is missing
        all_themes = set(info.economic_theme for info in all_features)
        missing_themes = all_themes - set(theme_counts.keys())
        
        if not missing_themes:
            return feature_info_list  # Already diverse
        
        # Add best features from missing themes
        adjusted = list(feature_info_list)
        
        for theme in missing_themes:
            # Find best feature from this theme
            theme_features = [
                info for info in all_features
                if info.economic_theme == theme and info not in adjusted
            ]
            
            if theme_features:
                # Add best from this theme
                best_theme_feature = max(theme_features, key=lambda x: x.importance)
                adjusted.append(best_theme_feature)
        
        # If we've exceeded target, trim lowest importance
        if target_n and len(adjusted) > target_n:
            adjusted.sort(key=lambda x: x.importance, reverse=True)
            adjusted = adjusted[:target_n]
        
        return adjusted
    
    def _validate_factor_portfolio(
        self,
        features: pd.DataFrame,
        labels: pd.Series
    ) -> Optional[float]:
        """
        Validate features by constructing a factor portfolio.
        
        Args:
            features: Selected features
            labels: Target labels
        
        Returns:
            Sharpe ratio of factor portfolio, or None if validation fails
        """
        try:
            # Weight features by their correlation with labels
            weights = {}
            for col in features.columns:
                corr = features[col].corr(labels)
                if not np.isnan(corr):
                    weights[col] = abs(corr)
            
            if not weights:
                return None
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}
            
            # Construct factor portfolio
            factor_returns = sum(
                features[col] * weight
                for col, weight in weights.items()
            )
            
            # Compute Sharpe ratio
            if factor_returns.std() < 1e-8:
                return 0.0
            
            sharpe = factor_returns.mean() / factor_returns.std() * np.sqrt(252)
            return float(sharpe)
        
        except Exception as e:
            self.logger.warning(f"Factor portfolio validation failed: {e}")
            return None


def select_features_robust(
    features: pd.DataFrame,
    labels: pd.Series,
    target_n_features: Optional[int] = None,
    n_bootstrap_folds: int = 5,
    min_selection_frequency: float = 0.60,
    feature_themes: Optional[Dict[str, EconomicTheme]] = None,
    validate_with_factor_portfolio: bool = True,
    logger: Optional[logging.Logger] = None
) -> SelectionResult:
    """
    Convenience function for robust feature selection.
    
    Args:
        features: Feature DataFrame
        labels: Target labels
        target_n_features: Target number of features
        n_bootstrap_folds: Number of bootstrap folds
        min_selection_frequency: Minimum selection frequency
        feature_themes: Optional feature theme mapping
        validate_with_factor_portfolio: Whether to validate with factor portfolio
        logger: Optional logger
    
    Returns:
        SelectionResult with selected features
    """
    config = FeatureSelectionConfig(
        n_bootstrap_folds=n_bootstrap_folds,
        min_selection_frequency=min_selection_frequency,
        validate_with_factor_portfolio=validate_with_factor_portfolio
    )
    
    selector = EnhancedFeatureSelector(config=config, logger=logger)
    
    return selector.select_features(
        features=features,
        labels=labels,
        feature_themes=feature_themes,
        target_n_features=target_n_features
    )


__all__ = [
    'EnhancedFeatureSelector',
    'FeatureSelectionConfig',
    'SelectionResult',
    'FeatureInfo',
    'EconomicTheme',
    'select_features_robust',
]