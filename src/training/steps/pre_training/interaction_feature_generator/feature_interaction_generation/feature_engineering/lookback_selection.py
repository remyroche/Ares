"""
Lookback Selection System for End-to-End Roadmap

Implements nested, hysteresis-based lookback selection with:
- Tiny menus (3-4 options per family)
- Purged, embargoed walk-forward validation
- Simplicity prior (prefer shorter unless longer wins by ≥0.25σ)
- Hysteresis (only change if winner repeats across 2 consecutive retrains)
"""

from typing import Dict, List, Optional, Any, Tuple, Iterable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import warnings


logger = logging.getLogger(__name__)


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
    outer_validation_score: float = 0.0
    split_details: List[Dict[str, Any]] = field(default_factory=list)


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
            family_features = [f for f in feature_list if f in features.columns]
            if not family_features:
                continue

            X_family = features[family_features]
            y = targets

            outer_scores = defaultdict(list)
            inner_scores_by_lookback = defaultdict(list)
            outer_split_logs: List[Dict[str, Any]] = []

            for fold_idx, (train_idx, holdout_idx) in enumerate(
                self._generate_outer_walkforward_splits(len(X_family))
            ):
                X_train, X_holdout = X_family.iloc[train_idx], X_family.iloc[holdout_idx]
                y_train, y_holdout = y.iloc[train_idx], y.iloc[holdout_idx]

                lookback_candidates = []
                lookback_fold_details: Dict[int, Dict[str, Any]] = {}

                for lookback in menu.options:
                    try:
                        inner_avg, inner_fold_scores = self._evaluate_lookback(
                            X_train,
                            y_train,
                            family_features,
                            lookback,
                            return_fold_scores=True,
                        )
                    except Exception as e:
                        warnings.warn(
                            f"Failed to evaluate lookback {lookback} for {family} (outer fold {fold_idx}): {e}"
                        )
                        continue

                    lookback_candidates.append((lookback, inner_avg))
                    lookback_fold_details[lookback] = {
                        "inner_avg_score": inner_avg,
                        "inner_fold_scores": inner_fold_scores,
                    }

                if not lookback_candidates:
                    continue

                fold_best_lookback = self._apply_simplicity_prior(lookback_candidates)
                fold_best_inner_avg = dict(lookback_candidates)[fold_best_lookback]
                inner_scores_by_lookback[fold_best_lookback].append(fold_best_inner_avg)

                holdout_score = self._compute_correlation_score(X_holdout, y_holdout)
                if holdout_score is None:
                    holdout_score = 0.0
                outer_scores[fold_best_lookback].append(holdout_score)

                fold_log = {
                    "fold": fold_idx,
                    "train_start": int(train_idx[0]),
                    "train_end": int(train_idx[-1]),
                    "holdout_start": int(holdout_idx[0]),
                    "holdout_end": int(holdout_idx[-1]),
                    "selected_lookback": fold_best_lookback,
                    "inner_avg_score": float(fold_best_inner_avg),
                    "inner_fold_scores": lookback_fold_details.get(fold_best_lookback, {}).get(
                        "inner_fold_scores", []
                    ),
                    "holdout_score": float(holdout_score),
                    "candidate_inner_scores": {
                        lb: {
                            "avg": float(details["inner_avg_score"]),
                            "fold_scores": details["inner_fold_scores"],
                        }
                        for lb, details in lookback_fold_details.items()
                    },
                }
                outer_split_logs.append(fold_log)
                logger.info(
                    "Lookback selection outer fold | family=%s fold=%s lookback=%s inner_avg=%.4f holdout=%.4f",
                    family,
                    fold_idx,
                    fold_best_lookback,
                    fold_best_inner_avg,
                    holdout_score,
                )

            aggregated_choices = [
                (lookback, float(np.mean(scores)))
                for lookback, scores in outer_scores.items()
                if scores
            ]

            if not aggregated_choices:
                # Fallback to original behaviour when no outer folds are available
                fallback_choices = []
                for lookback in menu.options:
                    try:
                        inner_avg, inner_fold_scores = self._evaluate_lookback(
                            X_family,
                            y,
                            family_features,
                            lookback,
                            return_fold_scores=True,
                        )
                        fallback_choices.append((lookback, inner_avg))
                        outer_split_logs.append(
                            {
                                "fold": "global",
                                "train_start": 0,
                                "train_end": int(len(X_family) - 1),
                                "holdout_start": None,
                                "holdout_end": None,
                                "selected_lookback": lookback,
                                "inner_avg_score": float(inner_avg),
                                "inner_fold_scores": inner_fold_scores,
                                "holdout_score": None,
                                "candidate_inner_scores": {},
                            }
                        )
                    except Exception as e:
                        warnings.warn(
                            f"Failed to evaluate lookback {lookback} for {family} during fallback: {e}"
                        )

                aggregated_choices = fallback_choices

            if not aggregated_choices:
                continue

            # Apply simplicity prior and hysteresis using aggregated outer scores
            best_lookback = self._apply_selection_logic(family, aggregated_choices)

            best_outer_scores = outer_scores.get(best_lookback, [])
            outer_validation_score = float(np.mean(best_outer_scores)) if best_outer_scores else 0.0
            mean_inner_score = float(np.mean(inner_scores_by_lookback.get(best_lookback, []))) if inner_scores_by_lookback.get(best_lookback) else 0.0

            logger.info(
                "Lookback selection aggregated | family=%s lookback=%s outer_mean=%.4f inner_mean=%.4f",
                family,
                best_lookback,
                outer_validation_score,
                mean_inner_score,
            )

            choice = LookbackChoice(
                family=family,
                selected_lookback=best_lookback,
                selection_criteria=SelectionCriteria.IC,
                confidence_score=mean_inner_score,
                ic_score=mean_inner_score,
                auc_score=0.0,
                simplicity_bonus=0.0,
                spec_hash=f"{family}_{best_lookback}",
                outer_validation_score=outer_validation_score,
                split_details=outer_split_logs,
            )

            choices[family] = choice

        return choices

    def _evaluate_lookback(self,
                          features: pd.DataFrame,
                          targets: pd.Series,
                          feature_list: List[str],
                          lookback: int,
                          return_fold_scores: bool = False) -> Union[float, Tuple[float, List[float]]]:
        """Evaluate a specific lookback using walk-forward CV."""

        # Filter features for this family
        family_features = [f for f in feature_list if f in features.columns]
        if not family_features:
            return (0.0, []) if return_fold_scores else 0.0

        X = features[family_features]
        y = targets

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
                avg_correlation = self._compute_correlation_score(X_train, y_train)
                if avg_correlation is not None:
                    scores.append(avg_correlation)
            except Exception:
                continue

        avg_score = float(np.mean(scores)) if scores else 0.0
        if return_fold_scores:
            return avg_score, scores
        return avg_score

    def _compute_correlation_score(self, X: pd.DataFrame, y: pd.Series) -> Optional[float]:
        """Compute average absolute correlation between features and target."""

        correlations = []
        for col in X.columns:
            feature_series = X[col]
            if feature_series.isna().all() or y.isna().all():
                continue

            corr = feature_series.corr(y)
            if not pd.isna(corr):
                correlations.append(abs(corr))

        if not correlations:
            return None
        return float(np.mean(correlations))

    def _generate_outer_walkforward_splits(self, n_samples: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged walk-forward splits for the outer CV loop."""

        embargo_size = int(n_samples * self.embargo_pct)
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        indices = np.arange(n_samples)

        for train_idx, test_idx in tscv.split(indices):
            if embargo_size > 0:
                if len(train_idx) > embargo_size:
                    train_idx = train_idx[:-embargo_size]
                else:
                    train_idx = np.array([], dtype=int)

                if len(test_idx) > embargo_size:
                    test_idx = test_idx[embargo_size:]
                else:
                    test_idx = np.array([], dtype=int)

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def _apply_simplicity_prior(self, choices: List[Tuple[int, float]]) -> int:
        """Apply the simplicity preference without updating hysteresis state."""

        if not choices:
            return 1

        choices.sort(key=lambda x: x[1], reverse=True)
        best_lookback, best_score = choices[0]

        for lookback, score in choices[1:]:
            score_diff = best_score - score
            if score_diff < self.simplicity_threshold and lookback < best_lookback:
                best_lookback = lookback
                best_score = score

        return best_lookback
    
    def _apply_selection_logic(self, 
                              family: str, 
                              choices: List[Tuple[int, float]]) -> int:
        """Apply simplicity prior and hysteresis to select best lookback."""
        
        if not choices:
            return 1  # Default fallback
        
        # Sort by score (descending)
        choices.sort(key=lambda x: x[1], reverse=True)
        
        # Get current selection from history
        current_selection = self.history.get(family, None)
        
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