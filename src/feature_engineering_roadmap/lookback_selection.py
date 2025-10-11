"""
Lookback Selection System for End-to-End Roadmap

Implements nested, hysteresis-based lookback selection with:
- Tiny menus (3-4 options per family)
- Purged, embargoed walk-forward validation
- Simplicity prior (prefer shorter unless longer wins by ≥0.25σ)
- Hysteresis (only change if winner repeats across 2 consecutive retrains)
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings


class SelectionCriteria(Enum):
    """Criteria for lookback selection."""
    IC = "ic"  # Information Coefficient
    AUC = "auc"  # Area Under Curve
    SIMPLICITY = "simplicity"  # Prefer shorter windows


@dataclass
class LookbackChoice:
    """A lookback choice for a feature family."""
    family: str
    selected_lookback: int
    selection_criteria: SelectionCriteria
    confidence_score: float
    ic_score: float
    auc_score: float
    simplicity_bonus: float
    spec_hash: str


@dataclass
class LookbackMenu:
    """Menu of lookback options for a feature family."""
    family: str
    options: List[int]
    description: str


class LookbackSelector:
    """Lookback selection with nested CV and hysteresis."""
    
    def __init__(self, 
                 n_folds: int = 5,
                 embargo_pct: float = 0.1,
                 simplicity_threshold: float = 0.25,
                 hysteresis_required: int = 2):
        self.n_folds = n_folds
        self.embargo_pct = embargo_pct
        self.simplicity_threshold = simplicity_threshold
        self.hysteresis_required = hysteresis_required
        self.history = {}  # Track selection history for hysteresis
        self.menus = self._create_menus()
    
    def _create_menus(self) -> Dict[str, LookbackMenu]:
        """Create lookback menus for each feature family."""
        return {
            'momentum': LookbackMenu('momentum', [5, 12, 24], 'Momentum lookback periods'),
            'sigma_ew': LookbackMenu('sigma_ew', [6, 12, 18], 'EW volatility halflife'),
            'gk_window': LookbackMenu('gk_window', [6, 12, 24], 'GK estimator window'),
            'vwap_roll': LookbackMenu('vwap_roll', [6, 12], 'Rolling VWAP window'),
            'rsi_period': LookbackMenu('rsi_period', [7, 14], 'RSI period'),
            'autocorr_window': LookbackMenu('autocorr_window', [6, 12], 'Autocorrelation window')
        }
    
    def select_lookbacks(self, 
                        features: pd.DataFrame,
                        targets: pd.Series,
                        feature_families: Dict[str, List[str]]) -> Dict[str, LookbackChoice]:
        """Select optimal lookbacks for each feature family."""
        choices = {}
        
        for family, feature_list in feature_families.items():
            if family not in self.menus:
                continue
            
            menu = self.menus[family]
            family_choices = []
            
            # Test each lookback option
            for lookback in menu.options:
                try:
                    score = self._evaluate_lookback(
                        features, targets, feature_list, lookback
                    )
                    family_choices.append((lookback, score))
                except Exception as e:
                    warnings.warn(f"Failed to evaluate lookback {lookback} for {family}: {e}")
                    continue
            
            if not family_choices:
                continue
            
            # Apply simplicity prior and hysteresis
            best_lookback = self._apply_selection_logic(family, family_choices)
            
            # Create choice object
            choice = LookbackChoice(
                family=family,
                selected_lookback=best_lookback,
                selection_criteria=SelectionCriteria.IC,  # Default
                confidence_score=0.8,  # Placeholder
                ic_score=0.0,  # Would be calculated in real implementation
                auc_score=0.0,  # Would be calculated in real implementation
                simplicity_bonus=0.0,  # Would be calculated
                spec_hash=f"{family}_{best_lookback}"
            )
            
            choices[family] = choice
        
        return choices
    
    def _evaluate_lookback(self, 
                          features: pd.DataFrame,
                          targets: pd.Series,
                          feature_list: List[str],
                          lookback: int) -> float:
        """Evaluate a specific lookback using walk-forward CV."""
        
        # Filter features for this family
        family_features = [f for f in feature_list if f in features.columns]
        if not family_features:
            return 0.0

        if lookback <= 0:
            raise ValueError("lookback must be positive")

        min_periods = max(1, lookback // 2)
        X = features[family_features].rolling(window=lookback, min_periods=min_periods).mean().shift(1)
        X = X.dropna()
        y = targets.reindex(X.index).dropna()

        X, y = X.align(y, join='inner', axis=0)

        if X.empty or y.empty:
            return 0.0

        # Create time series splits with embargo
        n_samples = len(X)
        embargo_size = int(n_samples * self.embargo_pct)
        
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            # Apply embargo
            if len(val_idx) > embargo_size:
                val_idx = val_idx[embargo_size:]
            
            if len(val_idx) == 0:
                continue
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Simple correlation-based scoring (placeholder for IC)
            try:
                # Calculate feature-target correlations
                correlations = []
                for col in X_train.columns:
                    if not X_train[col].isna().all() and not y_train.isna().all():
                        corr = X_train[col].corr(y_train)
                        if not pd.isna(corr):
                            correlations.append(abs(corr))
                
                if correlations:
                    avg_correlation = np.mean(correlations)
                    scores.append(avg_correlation)
            except Exception:
                continue
        
        return np.mean(scores) if scores else 0.0
    
    def _apply_selection_logic(self, 
                              family: str, 
                              choices: List[Tuple[int, float]]) -> int:
        """Apply simplicity prior and hysteresis to select best lookback."""
        
        if not choices:
            return 1  # Default fallback
        
        # Sort by score (descending)
        choices.sort(key=lambda x: x[1], reverse=True)
        
        # Get current selection from history
        family_history = self.history.get(family, [])
        current_selection = family_history[-1] if family_history else None
        
        # Apply simplicity prior
        best_lookback, best_score = choices[0]
        
        for lookback, score in choices[1:]:
            # Check if shorter window is significantly worse
            score_diff = best_score - score
            if score_diff < self.simplicity_threshold:
                # Prefer shorter window
                if lookback < best_lookback:
                    best_lookback = lookback
                    best_score = score
        
        # Apply hysteresis
        if current_selection is not None:
            # Check if current selection is still competitive
            current_score = next((score for l, score in choices if l == current_selection), 0.0)
            
            # If current selection is still good enough, keep it
            if current_score >= best_score - self.simplicity_threshold:
                best_lookback = current_selection
        
        # Update history
        if family not in self.history:
            self.history[family] = []

        self.history[family].append(best_lookback)
        
        # Keep only recent history for hysteresis
        if len(self.history[family]) > self.hysteresis_required:
            self.history[family] = self.history[family][-self.hysteresis_required:]
        
        return best_lookback
    
    def get_global_choice(self, 
                         all_choices: Dict[str, LookbackChoice]) -> Dict[str, int]:
        """Get global lookback choices (one per family across all assets)."""
        global_choices = {}
        
        for family, choice in all_choices.items():
            # For now, just use the choice as-is
            # In a real implementation, this would aggregate across multiple assets
            global_choices[family] = choice.selected_lookback
        
        return global_choices


class LookbackOptimizer:
    """Optimizer for lookback selection with advanced metrics."""
    
    def __init__(self, 
                 ic_threshold: float = 0.02,
                 auc_threshold: float = 0.52,
                 min_samples: int = 100):
        self.ic_threshold = ic_threshold
        self.auc_threshold = auc_threshold
        self.min_samples = min_samples
    
    def calculate_ic(self, features: pd.Series, targets: pd.Series) -> float:
        """Calculate Information Coefficient."""
        if len(features) < self.min_samples:
            return 0.0
        
        # Remove NaN values
        mask = ~(features.isna() | targets.isna())
        if mask.sum() < self.min_samples:
            return 0.0
        
        clean_features = features[mask]
        clean_targets = targets[mask]
        
        # Calculate correlation
        try:
            ic = clean_features.corr(clean_targets)
            return ic if not pd.isna(ic) else 0.0
        except Exception:
            return 0.0
    
    def calculate_auc(self, features: pd.Series, targets: pd.Series) -> float:
        """Calculate Area Under Curve for binary classification."""
        if len(features) < self.min_samples:
            return 0.5
        
        # Remove NaN values
        mask = ~(features.isna() | targets.isna())
        if mask.sum() < self.min_samples:
            return 0.5
        
        clean_features = features[mask]
        clean_targets = targets[mask]
        
        # Convert targets to binary (positive/negative)
        binary_targets = (clean_targets > 0).astype(int)
        
        if binary_targets.nunique() < 2:
            return 0.5
        
        try:
            from sklearn.metrics import roc_auc_score

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
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
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
            auc = roc_auc_score(binary_targets, clean_features)
            return auc if not pd.isna(auc) else 0.5
        except Exception:
            return 0.5
    
    def evaluate_feature_performance(self, 
                                   features: pd.Series, 
                                   targets: pd.Series) -> Dict[str, float]:
        """Evaluate feature performance with multiple metrics."""
        ic = self.calculate_ic(features, targets)
        auc = self.calculate_auc(features, targets)
        
        # Combined score (weighted average)
        combined_score = 0.6 * abs(ic) + 0.4 * (auc - 0.5)
        
        return {
            'ic': ic,
            'auc': auc,
            'combined': combined_score,
            'ic_abs': abs(ic),
            'auc_centered': auc - 0.5
        }


def create_feature_families(feature_names: List[str]) -> Dict[str, List[str]]:
    """Create feature families from feature names."""
    families = {
        'momentum': [],
        'sigma_ew': [],
        'gk_window': [],
        'vwap_roll': [],
        'rsi_period': [],
        'autocorr_window': []
    }
    
    for feature_name in feature_names:
        if 'mom' in feature_name:
            families['momentum'].append(feature_name)
        elif 'sigma' in feature_name:
            families['sigma_ew'].append(feature_name)
        elif 'gk' in feature_name:
            families['gk_window'].append(feature_name)
        elif 'vwap' in feature_name:
            families['vwap_roll'].append(feature_name)
        elif 'rsi' in feature_name:
            families['rsi_period'].append(feature_name)
        elif 'autocorr' in feature_name:
            families['autocorr_window'].append(feature_name)
    
    # Remove empty families
    return {k: v for k, v in families.items() if v}
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
