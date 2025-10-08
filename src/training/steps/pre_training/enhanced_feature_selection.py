"""
Enhanced Feature Selection Module

This module provides robust feature selection with:
1. Bootstrap stability testing
2. Economic interpretability layer
3. Feature grouping by economic theme
4. Out-of-sample IC tracking
5. Factor portfolio backtesting
6. Orthogonalization

Addresses Section 5: Feature Selection Stage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.linalg import orth

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


logger = logging.getLogger(__name__)


@dataclass
class EconomicTheme:
    """Economic theme for feature grouping."""
    
    theme_name: str
    description: str
    keywords: List[str]
    min_features: int = 1  # Minimum features to retain from this theme


# Define standard economic themes
STANDARD_THEMES = [
    EconomicTheme(
        theme_name="trend",
        description="Trend-following indicators",
        keywords=["ma", "ema", "trend", "adx", "momentum"],
        min_features=2
    ),
    EconomicTheme(
        theme_name="momentum",
        description="Momentum and rate of change",
        keywords=["roc", "mom", "rsi", "stoch", "momentum"],
        min_features=2
    ),
    EconomicTheme(
        theme_name="volatility",
        description="Volatility and dispersion measures",
        keywords=["std", "atr", "volatility", "bbands", "keltner"],
        min_features=2
    ),
    EconomicTheme(
        theme_name="volume",
        description="Volume and liquidity indicators",
        keywords=["volume", "obv", "vwap", "mfi"],
        min_features=1
    ),
    EconomicTheme(
        theme_name="microstructure",
        description="Market microstructure signals",
        keywords=["spread", "depth", "imbalance", "tick"],
        min_features=1
    ),
]


@dataclass
class FeatureSelectionResult:
    """Result of feature selection."""
    
    selected_features: List[str]
    feature_importances: Dict[str, float]
    
    # Stability metrics
    selection_frequency: Dict[str, float]  # How often feature selected across runs
    stable_features: List[str]  # Features appearing in >60-70% of runs
    
    # Economic grouping
    features_by_theme: Dict[str, List[str]]
    theme_coverage: Dict[str, int]
    
    # IC tracking
    feature_ic: Dict[str, float]
    feature_ic_tstat: Dict[str, float]
    
    # Orthogonalization
    orthogonalized: bool = False
    orthogonal_features: Optional[pd.DataFrame] = None
    
    # Metadata
    n_bootstrap_runs: int = 0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large DataFrames)."""
        return {
            'selected_features': self.selected_features,
            'feature_importances': self.feature_importances,
            'selection_frequency': self.selection_frequency,
            'stable_features': self.stable_features,
            'features_by_theme': self.features_by_theme,
            'theme_coverage': self.theme_coverage,
            'feature_ic': self.feature_ic,
            'feature_ic_tstat': self.feature_ic_tstat,
            'orthogonalized': self.orthogonalized,
            'n_bootstrap_runs': self.n_bootstrap_runs,
            'timestamp': self.timestamp
        }


class EnhancedFeatureSelector:
    """
    Enhanced feature selector with economic interpretability and robustness.
    
    Key Features:
    1. Bootstrap-based stability testing
    2. Economic theme grouping
    3. IC tracking and validation
    4. Factor portfolio backtesting
    5. Feature orthogonalization
    """
    
    def __init__(
        self,
        themes: Optional[List[EconomicTheme]] = None,
        stability_threshold: float = 0.6,
        min_ic: float = 0.01,
        min_ic_tstat: float = 2.0
    ):
        """
        Initialize enhanced feature selector.
        
        Args:
            themes: Economic themes for grouping
            stability_threshold: Minimum frequency for stable features
            min_ic: Minimum information coefficient
            min_ic_tstat: Minimum IC t-statistic
        """
        self.themes = themes or STANDARD_THEMES
        self.stability_threshold = stability_threshold
        self.min_ic = min_ic
        self.min_ic_tstat = min_ic_tstat
        
        tprint_success("✅ EnhancedFeatureSelector initialized")
        tprint_info(f"   → Economic themes: {len(self.themes)}")
        tprint_info(f"   → Stability threshold: {self.stability_threshold:.1%}")
        tprint_info(f"   → Min IC: {self.min_ic}")
    
    def select_features_with_bootstrap(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_bootstrap: int = 20,
        subsample_ratio: float = 0.8,
        max_features: int = 100
    ) -> FeatureSelectionResult:
        """
        Select features using bootstrap stability testing.
        
        Args:
            X: Feature DataFrame
            y: Target series
            n_bootstrap: Number of bootstrap iterations
            subsample_ratio: Ratio of samples to use in each iteration
            max_features: Maximum features to select
        
        Returns:
            FeatureSelectionResult
        """
        tprint_info(f"🔄 Performing bootstrap feature selection ({n_bootstrap} runs)...")
        
        # Track feature selection across runs
        feature_selection_count = {col: 0 for col in X.columns}
        feature_importance_sum = {col: 0.0 for col in X.columns}
        
        for run in range(n_bootstrap):
            # Bootstrap sample
            sample_size = int(len(X) * subsample_ratio)
            sample_idx = np.random.choice(len(X), size=sample_size, replace=True)
            
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
            
            # Fit feature selector (Random Forest for importance)
            selector = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=run,
                n_jobs=-1
            )
            
            selector.fit(X_sample.fillna(0), y_sample)
            
            # Get feature importances
            importances = selector.feature_importances_
            
            # Select top features
            top_indices = np.argsort(importances)[-max_features:]
            
            for idx in top_indices:
                feature_name = X.columns[idx]
                feature_selection_count[feature_name] += 1
                feature_importance_sum[feature_name] += importances[idx]
        
        # Calculate selection frequency
        selection_frequency = {
            feature: count / n_bootstrap
            for feature, count in feature_selection_count.items()
        }
        
        # Average importance
        feature_importances = {
            feature: imp / n_bootstrap
            for feature, imp in feature_importance_sum.items()
        }
        
        # Identify stable features (appear in >stability_threshold% of runs)
        stable_features = [
            feature for feature, freq in selection_frequency.items()
            if freq >= self.stability_threshold
        ]
        
        # Calculate IC for stable features
        feature_ic, feature_ic_tstat = self._calculate_ic_metrics(
            X[stable_features], y
        )
        
        # Filter by IC
        ic_filtered_features = [
            feature for feature in stable_features
            if feature_ic.get(feature, 0) >= self.min_ic
            and feature_ic_tstat.get(feature, 0) >= self.min_ic_tstat
        ]
        
        # Group by economic theme
        features_by_theme, theme_coverage = self._group_by_theme(ic_filtered_features)
        
        # Ensure minimum coverage per theme
        final_features = self._ensure_theme_coverage(
            ic_filtered_features, features_by_theme, X, y
        )
        
        tprint_success(f"✅ Bootstrap selection complete: {len(final_features)} features")
        tprint_info(f"   → Stable features: {len(stable_features)}")
        tprint_info(f"   → IC-filtered: {len(ic_filtered_features)}")
        tprint_info(f"   → Theme coverage: {list(theme_coverage.values())}")
        
        return FeatureSelectionResult(
            selected_features=final_features,
            feature_importances=feature_importances,
            selection_frequency=selection_frequency,
            stable_features=stable_features,
            features_by_theme=features_by_theme,
            theme_coverage=theme_coverage,
            feature_ic=feature_ic,
            feature_ic_tstat=feature_ic_tstat,
            n_bootstrap_runs=n_bootstrap
        )
    
    def _calculate_ic_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate information coefficient and t-statistic for each feature.
        
        Args:
            X: Feature DataFrame
            y: Target series
        
        Returns:
            Tuple of (IC dict, IC t-stat dict)
        """
        feature_ic = {}
        feature_ic_tstat = {}
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        for col in X.columns:
            feature = X_aligned[col].dropna()
            target = y_aligned.loc[feature.index]
            
            if len(feature) > 10:
                # Calculate Spearman IC
                ic = feature.corr(target, method='spearman')
                
                # Calculate t-statistic
                n = len(feature)
                if not np.isnan(ic) and n > 2:
                    t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-8)
                    feature_ic[col] = float(ic)
                    feature_ic_tstat[col] = float(t_stat)
        
        return feature_ic, feature_ic_tstat
    
    def _group_by_theme(
        self,
        features: List[str]
    ) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """
        Group features by economic theme.
        
        Args:
            features: List of feature names
        
        Returns:
            Tuple of (features by theme, theme coverage counts)
        """
        features_by_theme = {theme.theme_name: [] for theme in self.themes}
        
        for feature in features:
            feature_lower = feature.lower()
            
            # Check which theme this feature belongs to
            assigned = False
            for theme in self.themes:
                if any(keyword in feature_lower for keyword in theme.keywords):
                    features_by_theme[theme.theme_name].append(feature)
                    assigned = True
                    break
            
            # If no theme matches, add to "other"
            if not assigned:
                if "other" not in features_by_theme:
                    features_by_theme["other"] = []
                features_by_theme["other"].append(feature)
        
        # Count coverage
        theme_coverage = {
            theme: len(features)
            for theme, features in features_by_theme.items()
        }
        
        return features_by_theme, theme_coverage
    
    def _ensure_theme_coverage(
        self,
        features: List[str],
        features_by_theme: Dict[str, List[str]],
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[str]:
        """
        Ensure minimum coverage per economic theme.
        
        Args:
            features: Current feature list
            features_by_theme: Features grouped by theme
            X: Feature DataFrame
            y: Target series
        
        Returns:
            Updated feature list with theme coverage
        """
        final_features = list(features)
        
        for theme in self.themes:
            theme_features = features_by_theme.get(theme.theme_name, [])
            
            if len(theme_features) < theme.min_features:
                # Need to add more features from this theme
                needed = theme.min_features - len(theme_features)
                
                # Find candidate features from this theme
                candidates = [
                    col for col in X.columns
                    if any(keyword in col.lower() for keyword in theme.keywords)
                    and col not in final_features
                ]
                
                if candidates:
                    # Select top candidates by IC
                    ic_values = {}
                    for col in candidates:
                        feature = X[col].dropna()
                        target = y.loc[feature.index]
                        if len(feature) > 10:
                            ic = feature.corr(target, method='spearman')
                            if not np.isnan(ic):
                                ic_values[col] = abs(ic)
                    
                    # Add top candidates
                    if ic_values:
                        sorted_candidates = sorted(ic_values.items(), key=lambda x: x[1], reverse=True)
                        for col, _ in sorted_candidates[:needed]:
                            final_features.append(col)
                            tprint_info(f"   → Added {col} for theme coverage: {theme.theme_name}")
        
        return final_features
    
    def orthogonalize_features(
        self,
        X: pd.DataFrame,
        selected_features: List[str]
    ) -> pd.DataFrame:
        """
        Orthogonalize selected features to reduce multicollinearity.
        
        Args:
            X: Feature DataFrame
            selected_features: List of selected features
        
        Returns:
            Orthogonalized feature DataFrame
        """
        tprint_info("🔄 Orthogonalizing features...")
        
        # Select features
        X_selected = X[selected_features].fillna(0)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # Orthogonalize using QR decomposition
        X_orth = orth(X_scaled)
        
        # Create DataFrame with orthogonal features
        orth_feature_names = [f'orth_{i}' for i in range(X_orth.shape[1])]
        X_orthogonal = pd.DataFrame(
            X_orth,
            index=X_selected.index,
            columns=orth_feature_names
        )
        
        tprint_success(f"✅ Orthogonalized {X_selected.shape[1]} features")
        
        return X_orthogonal
    
    def backtest_factor_portfolio(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Backtest a factor portfolio built from selected features.
        
        Args:
            X: Feature DataFrame
            y: Target series (returns)
            selected_features: List of selected features
            weights: Optional feature weights
        
        Returns:
            Dictionary with backtest results
        """
        tprint_info("📊 Backtesting factor portfolio...")
        
        # Use equal weights if not provided
        if weights is None:
            weights = {feature: 1.0 / len(selected_features) for feature in selected_features}
        
        # Align data
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx, selected_features]
        y_aligned = y.loc[common_idx]
        
        # Construct factor portfolio (weighted average of features)
        factor_portfolio = pd.Series(0.0, index=X_aligned.index)
        
        for feature, weight in weights.items():
            if feature in X_aligned.columns:
                factor_portfolio += X_aligned[feature].fillna(0) * weight
        
        # Calculate portfolio returns (assuming features are signals)
        # Normalize signals to [-1, 1]
        factor_portfolio = factor_portfolio / (factor_portfolio.abs().max() + 1e-8)
        
        # Calculate returns
        portfolio_returns = factor_portfolio * y_aligned
        
        # Calculate metrics
        total_return = portfolio_returns.sum()
        sharpe_ratio = (
            portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            if portfolio_returns.std() > 0 else 0
        )
        max_drawdown = self._calculate_max_drawdown(portfolio_returns.cumsum())
        
        result = {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'mean_return': float(portfolio_returns.mean()),
            'std_return': float(portfolio_returns.std()),
            'n_trades': int((factor_portfolio != 0).sum())
        }
        
        tprint_success(f"✅ Backtest complete: Sharpe={sharpe_ratio:.2f}, Total Return={total_return:.2%}")
        
        return result
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        max_drawdown = drawdown.min()
        return abs(max_drawdown)


def create_enhanced_feature_selector(
    themes: Optional[List[EconomicTheme]] = None,
    stability_threshold: float = 0.6,
    min_ic: float = 0.01,
    min_ic_tstat: float = 2.0
) -> EnhancedFeatureSelector:
    """
    Factory function to create EnhancedFeatureSelector.
    
    Args:
        themes: Economic themes
        stability_threshold: Stability threshold
        min_ic: Minimum IC
        min_ic_tstat: Minimum IC t-stat
    
    Returns:
        EnhancedFeatureSelector instance
    """
    return EnhancedFeatureSelector(themes, stability_threshold, min_ic, min_ic_tstat)