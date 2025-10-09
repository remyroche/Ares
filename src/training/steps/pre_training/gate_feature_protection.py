"""
Gate Feature Protection for Final Feature Selection

This module provides protection mechanisms to ensure gate features are not
filtered out during final_feature_selection, while maintaining data-driven
selection principles.

Key Features:
- Gate feature identification and protection
- Specialized selection criteria for gates
- Regime-aware gate validation
- Integration with existing selection pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import logging
from enum import Enum

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_warning, tprint_success, tprint_error


class GateFeatureType(Enum):
    """Types of gate features."""
    GATED_TWIN_POS = "gated_twin_pos"      # feature_pos = feature * (1 - p_fail)
    GATED_TWIN_NEG = "gated_twin_neg"      # feature_neg = -feature * p_fail
    EXCEPTION_INTERACTION = "exception_interaction"  # feature_x_fail = feature * p_fail
    CONTEXT_INDICATOR = "context_indicator"  # feature_p_context = p_fail
    REGIME_INTERACTION = "regime_interaction"  # feature_x_regime
    VOLATILITY_GATE = "volatility_gate"    # vol-scaled features
    LIQUIDITY_GATE = "liquidity_gate"      # spread-scaled features


@dataclass
class GateFeatureConfig:
    """Configuration for gate feature protection."""
    # Protection settings
    protect_gate_features: bool = True
    max_gate_features_per_base: int = 5  # Max gates per base feature (updated cap)
    min_gate_ic_improvement: float = 0.005  # Must improve IC by this amount
    min_gate_stability: float = 0.4  # Minimum stability score
    
    # Selection criteria for gates
    gate_correlation_threshold: float = 0.95  # Higher threshold for gates
    gate_importance_weight: float = 1.5  # Boost importance scores
    gate_regime_bonus: float = 0.1  # Bonus for regime separation
    
    # Validation settings
    validate_gate_contribution: bool = True
    min_gate_contribution: float = 0.01  # Minimum contribution to model
    enable_gate_interaction_validation: bool = True


class GateFeatureProtector:
    """
    Protects gate features during final feature selection.
    
    This class ensures that gate features are not filtered out by
    correlation filtering, RFE, or other selection methods, while
    maintaining data-driven selection principles.
    """
    
    def __init__(self, config: Optional[GateFeatureConfig] = None):
        self.config = config or GateFeatureConfig()
        self.logger = system_logger.getChild('GateFeatureProtector')
        
        # Gate feature patterns
        self.gate_patterns = {
            GateFeatureType.GATED_TWIN_POS: ['_pos', '_positive'],
            GateFeatureType.GATED_TWIN_NEG: ['_neg', '_negative'],
            GateFeatureType.EXCEPTION_INTERACTION: ['_x_fail', '_x_exception'],
            GateFeatureType.CONTEXT_INDICATOR: ['_p_', '_prob_', '_context_'],
            GateFeatureType.REGIME_INTERACTION: ['_x_highvol', '_x_widespread', '_x_chop'],
            GateFeatureType.VOLATILITY_GATE: ['_x_rv', '_x_vol', '_x_sigma'],
            GateFeatureType.LIQUIDITY_GATE: ['_x_spread', '_x_liquidity']
        }
        
        # Protected features
        self.protected_features: Set[str] = set()
        self.gate_feature_mapping: Dict[str, str] = {}  # gate -> base feature
    
    def identify_gate_features(self, features: pd.DataFrame) -> Dict[str, GateFeatureType]:
        """Identify gate features in the feature set."""
        gate_features = {}
        
        for feature_name in features.columns:
            for gate_type, patterns in self.gate_patterns.items():
                if any(pattern in feature_name.lower() for pattern in patterns):
                    gate_features[feature_name] = gate_type
                    
                    # Map to base feature
                    base_feature = self._extract_base_feature_name(feature_name, gate_type)
                    self.gate_feature_mapping[feature_name] = base_feature
                    break
        
        self.logger.info(f"Identified {len(gate_features)} gate features")
        return gate_features
    
    def _extract_base_feature_name(self, gate_feature: str, gate_type: GateFeatureType) -> str:
        """Extract base feature name from gate feature name."""
        # Remove gate-specific suffixes
        base_name = gate_feature
        
        for pattern in self.gate_patterns[gate_type]:
            if pattern in base_name:
                base_name = base_name.replace(pattern, '')
                break
        
        # Remove common prefixes
        prefixes_to_remove = ['gate_', 'gated_', 'protected_']
        for prefix in prefixes_to_remove:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
                break
        
        return base_name
    
    def protect_gate_features(
        self, 
        features: pd.DataFrame, 
        target: pd.Series,
        selection_method: str = "correlation_filtering"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Protect gate features during feature selection.
        
        Args:
            features: Feature matrix
            target: Target variable
            selection_method: Selection method being applied
            
        Returns:
            Tuple of (protected_features, protection_info)
        """
        if not self.config.protect_gate_features:
            return features, {}
        
        # Identify gate features
        gate_features = self.identify_gate_features(features)
        
        if not gate_features:
            self.logger.info("No gate features found, no protection needed")
            return features, {}
        
        # Separate gate and non-gate features
        gate_feature_names = list(gate_features.keys())
        non_gate_features = [col for col in features.columns if col not in gate_feature_names]
        
        self.logger.info(f"Protecting {len(gate_feature_names)} gate features")
        
        # Apply protection based on selection method
        if selection_method == "correlation_filtering":
            return self._protect_from_correlation_filtering(
                features, target, gate_feature_names, non_gate_features
            )
        elif selection_method == "rfe":
            return self._protect_from_rfe(
                features, target, gate_feature_names, non_gate_features
            )
        elif selection_method == "variance_filtering":
            return self._protect_from_variance_filtering(
                features, target, gate_feature_names, non_gate_features
            )
        else:
            return self._protect_generic(
                features, target, gate_feature_names, non_gate_features
            )
    
    def _protect_from_correlation_filtering(
        self, 
        features: pd.DataFrame, 
        target: pd.Series,
        gate_features: List[str],
        non_gate_features: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Protect gate features from correlation filtering."""
        
        # Apply correlation filtering to non-gate features
        non_gate_df = features[non_gate_features]
        filtered_non_gate = self._apply_correlation_filtering(
            non_gate_df, target, self.config.gate_correlation_threshold
        )
        
        # Validate gate features
        valid_gates = self._validate_gate_features(
            features[gate_features], target
        )
        
        # Combine protected features
        protected_features = pd.concat([
            filtered_non_gate,
            features[valid_gates]
        ], axis=1)
        
        protection_info = {
            'original_gate_count': len(gate_features),
            'valid_gate_count': len(valid_gates),
            'protected_gates': valid_gates,
            'filtered_non_gates': len(filtered_non_gate.columns)
        }
        
        return protected_features, protection_info
    
    def _protect_from_rfe(
        self, 
        features: pd.DataFrame, 
        target: pd.Series,
        gate_features: List[str],
        non_gate_features: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Protect gate features from RFE."""
        
        # Apply RFE to non-gate features
        non_gate_df = features[non_gate_features]
        rfe_selected = self._apply_rfe_selection(non_gate_df, target)
        
        # Validate gate features
        valid_gates = self._validate_gate_features(
            features[gate_features], target
        )
        
        # Combine protected features
        protected_features = pd.concat([
            rfe_selected,
            features[valid_gates]
        ], axis=1)
        
        protection_info = {
            'original_gate_count': len(gate_features),
            'valid_gate_count': len(valid_gates),
            'protected_gates': valid_gates,
            'rfe_selected_non_gates': len(rfe_selected.columns)
        }
        
        return protected_features, protection_info
    
    def _protect_from_variance_filtering(
        self, 
        features: pd.DataFrame, 
        target: pd.Series,
        gate_features: List[str],
        non_gate_features: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Protect gate features from variance filtering."""
        
        # Apply variance filtering to non-gate features
        non_gate_df = features[non_gate_features]
        filtered_non_gate = self._apply_variance_filtering(non_gate_df)
        
        # Gate features are exempt from variance filtering
        # (they can have low variance by design)
        valid_gates = self._validate_gate_features(
            features[gate_features], target
        )
        
        # Combine protected features
        protected_features = pd.concat([
            filtered_non_gate,
            features[valid_gates]
        ], axis=1)
        
        protection_info = {
            'original_gate_count': len(gate_features),
            'valid_gate_count': len(valid_gates),
            'protected_gates': valid_gates,
            'variance_filtered_non_gates': len(filtered_non_gate.columns)
        }
        
        return protected_features, protection_info
    
    def _protect_generic(
        self, 
        features: pd.DataFrame, 
        target: pd.Series,
        gate_features: List[str],
        non_gate_features: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generic protection for unknown selection methods."""
        
        # Validate gate features
        valid_gates = self._validate_gate_features(
            features[gate_features], target
        )
        
        # Keep all non-gate features (let other methods handle them)
        protected_features = features
        
        protection_info = {
            'original_gate_count': len(gate_features),
            'valid_gate_count': len(valid_gates),
            'protected_gates': valid_gates
        }
        
        return protected_features, protection_info
    
    def _validate_gate_features(
        self, 
        gate_features: pd.DataFrame, 
        target: pd.Series
    ) -> List[str]:
        """Validate gate features using specialized criteria."""
        valid_gates = []
        
        for gate_name in gate_features.columns:
            gate_series = gate_features[gate_name]
            
            # Skip if all NaN
            if gate_series.isna().all():
                continue
            
            # Calculate IC
            ic = gate_series.corr(target)
            if pd.isna(ic):
                continue
            
            # Check IC improvement over base feature
            base_feature_name = self.gate_feature_mapping.get(gate_name)
            if base_feature_name and base_feature_name in gate_features.columns:
                base_ic = gate_features[base_feature_name].corr(target)
                ic_improvement = abs(ic) - abs(base_ic)
                
                if ic_improvement < self.config.min_gate_ic_improvement:
                    continue
            
            # Check stability
            stability = self._calculate_gate_stability(gate_series, target)
            if stability < self.config.min_gate_stability:
                continue
            
            # Check contribution to model
            if self.config.validate_gate_contribution:
                contribution = self._calculate_gate_contribution(gate_series, target)
                if contribution < self.config.min_gate_contribution:
                    continue
            
            valid_gates.append(gate_name)
        
        self.logger.info(f"Validated {len(valid_gates)}/{len(gate_features.columns)} gate features")
        return valid_gates
    
    def _calculate_gate_stability(self, gate_series: pd.Series, target: pd.Series) -> float:
        """Calculate stability of gate feature."""
        try:
            # Rolling correlation
            window = min(100, len(gate_series) // 4)
            rolling_corr = gate_series.rolling(window).corr(target)
            
            # Stability = 1 - std(rolling_corr)
            stability = 1 - rolling_corr.std()
            return max(0, stability)  # Ensure non-negative
            
        except Exception:
            return 0.0
    
    def _calculate_gate_contribution(self, gate_series: pd.Series, target: pd.Series) -> float:
        """Calculate contribution of gate feature to model."""
        try:
            # Simple R² as contribution measure
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            # Remove NaN values
            valid_mask = ~(gate_series.isna() | target.isna())
            if valid_mask.sum() < 10:
                return 0.0
            
            X = gate_series[valid_mask].values.reshape(-1, 1)
            y = target[valid_mask].values
            
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            return max(0, r2)
            
        except Exception:
            return 0.0
    
    def _apply_correlation_filtering(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        threshold: float
    ) -> pd.DataFrame:
        """Apply correlation filtering to non-gate features."""
        if features.empty:
            return features
        
        # Calculate correlation matrix
        corr_matrix = features.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            # Keep the feature with higher IC
            ic1 = features[feat1].corr(target)
            ic2 = features[feat2].corr(target)
            
            if abs(ic1) < abs(ic2):
                features_to_remove.add(feat1)
            else:
                features_to_remove.add(feat2)
        
        # Return filtered features
        remaining_features = [col for col in features.columns if col not in features_to_remove]
        return features[remaining_features]
    
    def _apply_rfe_selection(
        self, 
        features: pd.DataFrame, 
        target: pd.Series
    ) -> pd.DataFrame:
        """Apply RFE selection to non-gate features."""
        if features.empty:
            return features
        
        try:
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestRegressor
            
            # Use RandomForest for RFE
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            
            # Select top 50% of features
            n_features = max(10, len(features.columns) // 2)
            selector = RFE(estimator, n_features_to_select=n_features)
            
            selector.fit(features, target)
            selected_features = features.columns[selector.support_]
            
            return features[selected_features]
            
        except Exception as e:
            self.logger.warning(f"RFE selection failed: {e}, keeping all features")
            return features
    
    def _apply_variance_filtering(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply variance filtering to non-gate features."""
        if features.empty:
            return features
        
        # Calculate variance
        variances = features.var()
        
        # Remove low variance features (bottom 10%)
        threshold = variances.quantile(0.1)
        high_variance_features = variances[variances > threshold].index
        
        return features[high_variance_features]
    
    def get_protection_summary(self) -> Dict[str, Any]:
        """Get summary of gate feature protection."""
        return {
            'protected_features': list(self.protected_features),
            'gate_feature_mapping': self.gate_feature_mapping,
            'config': self.config.__dict__
        }


class GateAwareFeatureSelector:
    """
    Feature selector that is aware of gate features and protects them.
    
    This class wraps existing feature selection methods to ensure
    gate features are not filtered out inappropriately.
    """
    
    def __init__(self, base_selector, gate_config: Optional[GateFeatureConfig] = None):
        self.base_selector = base_selector
        self.gate_protector = GateFeatureProtector(gate_config)
        self.logger = system_logger.getChild('GateAwareFeatureSelector')
    
    def select_features(
        self, 
        features: pd.DataFrame, 
        target: pd.Series,
        method: str = "correlation_filtering"
    ) -> pd.DataFrame:
        """Select features while protecting gate features."""
        
        # Protect gate features
        protected_features, protection_info = self.gate_protector.protect_gate_features(
            features, target, method
        )
        
        # Apply base selection to protected features
        selected_features = self.base_selector.select_features(
            protected_features, target
        )
        
        # Log protection results
        self.logger.info(f"Gate protection: {protection_info}")
        
        return selected_features


def create_gate_aware_selector(base_selector, gate_config: Optional[GateFeatureConfig] = None):
    """Factory function to create gate-aware feature selector."""
    return GateAwareFeatureSelector(base_selector, gate_config)