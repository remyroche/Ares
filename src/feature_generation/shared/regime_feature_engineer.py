"""
Shared Regime Feature Engineering Module

This module provides shared feature engineering utilities for regime detection
that ensure consistency between training and inference (live trading).

Features:
1. Generates features using the same feature bank system as training
2. Applies LGBM-filtered feature selection (same 60-80 features as training)
3. Provides base model outputs for ensemble models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass, field
import pickle
import json
from pathlib import Path


@dataclass
class RegimeFeatureEngineeringResult:
    """Result of regime feature engineering operation."""
    features: np.ndarray
    feature_names: List[str]
    selected_features: Optional[np.ndarray] = None
    selected_feature_names: Optional[List[str]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class RegimeFeatureEngineer:
    """
    Feature engineer for regime detection models.
    
    Generates features using the same feature bank system as training
    and applies LGBM-filtered feature selection.
    """
    
    def __init__(
        self,
        selected_feature_names: Optional[List[str]] = None,
        feature_selection_info: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize regime feature engineer.
        
        Args:
            selected_feature_names: List of selected feature names from training
            feature_selection_info: Feature selection metadata from training
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.selected_feature_names = selected_feature_names or []
        self.feature_selection_info = feature_selection_info or {}
        
        # Feature generation components (initialized on demand)
        self._feature_bank = None
        self._regime_generator = None
        
    def _get_feature_bank(self):
        """Get feature bank (lazy initialization)."""
        if self._feature_bank is None:
            try:
                from src.feature_generation.core.factory import get_feature_bank, FeatureCategory
                self._feature_bank = get_feature_bank()
                self._feature_categories = [
                    FeatureCategory.REGIME,
                    FeatureCategory.MOMENTUM,
                    FeatureCategory.VOLATILITY,
                    FeatureCategory.VOLUME,
                    FeatureCategory.TREND,
                    FeatureCategory.OSCILLATOR,
                    FeatureCategory.RETURNS
                ]
            except ImportError as e:
                self.logger.warning(f"Feature bank not available: {e}")
                return None
        return self._feature_bank
    
    def _get_regime_generator(self):
        """Get regime feature generator (lazy initialization)."""
        if self._regime_generator is None:
            try:
                from src.feature_generation.categories.regime_feature_integration import (
                    RegimeFeatureIntegration, RegimeFeatureConfig
                )
                regime_config = RegimeFeatureConfig(
                    enable_regime_detection=True,
                    enable_adaptive_features=True,
                    enable_regime_transitions=True
                )
                self._regime_generator = RegimeFeatureIntegration(regime_config)
            except ImportError as e:
                self.logger.warning(f"Regime feature generator not available: {e}")
                return None
        return self._regime_generator
    
    def generate_features(
        self,
        market_data: pd.DataFrame,
        apply_selection: bool = True
    ) -> RegimeFeatureEngineeringResult:
        """
        Generate features from market data using the same pipeline as training.
        
        Args:
            market_data: Market data DataFrame (OHLCV format)
            apply_selection: Whether to apply feature selection
            
        Returns:
            RegimeFeatureEngineeringResult with features and metadata
        """
        warnings = []
        errors = []
        
        try:
            # Generate features using feature bank (same as training)
            all_features, all_feature_names = self._generate_features_with_bank(market_data)
            
            if all_features is None or len(all_feature_names) == 0:
                error_msg = "Failed to generate features from feature bank"
                errors.append(error_msg)
                return RegimeFeatureEngineeringResult(
                    features=np.array([]),
                    feature_names=[],
                    errors=errors,
                    warnings=warnings
                )
            
            # Apply feature selection if requested and available
            selected_features = None
            selected_feature_names = None
            
            if apply_selection and self.selected_feature_names:
                selected_features, selected_feature_names = self._apply_feature_selection(
                    all_features, all_feature_names, self.selected_feature_names
                )
                
                if selected_features is None:
                    warnings.append("Feature selection failed, using all features")
                    selected_features = all_features
                    selected_feature_names = all_feature_names
            
            elif apply_selection:
                warnings.append("No selected feature names available, using all features")
                selected_features = all_features
                selected_feature_names = all_feature_names
            else:
                selected_features = all_features
                selected_feature_names = all_feature_names
            
            return RegimeFeatureEngineeringResult(
                features=all_features,
                feature_names=all_feature_names,
                selected_features=selected_features,
                selected_feature_names=selected_feature_names,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Feature generation failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            return RegimeFeatureEngineeringResult(
                features=np.array([]),
                feature_names=[],
                errors=errors,
                warnings=warnings
            )
    
    def _generate_features_with_bank(
        self,
        data: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """
        Generate features using feature bank (same as training).
        
        This replicates the logic in regime_models_training._generate_features_with_bank()
        """
        try:
            feature_bank = self._get_feature_bank()
            if feature_bank is None:
                self.logger.error("Feature bank not available")
                return None, None
            
            # Generate core regime features
            regime_generator = self._get_regime_generator()
            core_regime_features = pd.DataFrame(index=data.index)
            
            if regime_generator:
                try:
                    # Generate core regime features using rolling window
                    window_size = 20
                    all_feature_keys = set()
                    
                    # Sample windows to get feature names
                    sample_windows = [
                        data.iloc[:min(20, len(data))],
                        data.iloc[max(0, len(data)-20):] if len(data) > 20 else data
                    ]
                    for window_data in sample_windows:
                        if len(window_data) >= 5:
                            sample_features = regime_generator._generate_regime_features(window_data)
                            all_feature_keys.update(sample_features.keys())
                    
                    # Initialize DataFrame
                    for feature_name in all_feature_keys:
                        core_regime_features[feature_name] = np.nan
                    
                    # Generate features row by row
                    for i in range(len(data)):
                        if i < 5:
                            window_data = data.iloc[:i+1]
                        else:
                            window_start = max(0, i - window_size + 1)
                            window_data = data.iloc[window_start:i+1]
                        
                        regime_features_dict = regime_generator._generate_regime_features(window_data)
                        
                        for feature_name, feature_value in regime_features_dict.items():
                            if isinstance(feature_value, (int, float)):
                                core_regime_features.loc[data.index[i], feature_name] = feature_value
                            elif isinstance(feature_value, bool):
                                core_regime_features.loc[data.index[i], feature_name] = float(feature_value)
                            elif isinstance(feature_value, str):
                                core_regime_features.loc[data.index[i], feature_name] = hash(feature_value) % 1000
                            else:
                                core_regime_features.loc[data.index[i], feature_name] = 0.0
                                
                except Exception as e:
                    self.logger.warning(f"Core regime feature generation failed: {e}")
            
            # Generate features from feature bank categories
            all_features = pd.DataFrame(index=data.index)
            
            if not core_regime_features.empty:
                all_features = pd.concat([all_features, core_regime_features], axis=1)
            
            # Generate features for each category
            for category in getattr(self, '_feature_categories', []):
                try:
                    generators = feature_bank.get_generators_by_category(category)
                    if not generators:
                        continue
                    
                    category_features = pd.DataFrame(index=data.index)
                    
                    for generator in generators:
                        try:
                            result = generator.generate(data)
                            if result and hasattr(result, 'data') and not result.data.empty if hasattr(result.data, 'empty') else result.data is not None:
                                feature_name = f"{category.value}_{getattr(generator.config, 'name', 'feature')}"
                                if hasattr(result, 'data'):
                                    category_features[feature_name] = result.data
                                else:
                                    category_features[feature_name] = result
                        except Exception as e:
                            self.logger.debug(f"Generator {getattr(generator.config, 'name', 'unknown')} failed: {e}")
                            continue
                    
                    if not category_features.empty:
                        all_features = pd.concat([all_features, category_features], axis=1)
                        
                except Exception as e:
                    self.logger.warning(f"Category {category.value} feature generation failed: {e}")
                    continue
            
            # Convert to numpy array
            if not all_features.empty:
                X = all_features.values
                feature_names = list(all_features.columns)
                
                # Add smoothed features if enabled (matching training)
                try:
                    from src.utils.ml_common.feature_engineering.feature_smoothing import add_smoothed_features
                    X, feature_names = add_smoothed_features(
                        X,
                        window_sizes=[3, 5, 7],
                        feature_names=feature_names
                    )
                except Exception as e:
                    self.logger.debug(f"Smoothed features not available: {e}")
                
                return X, feature_names
            else:
                return None, None
                
        except Exception as e:
            self.logger.error(f"Feature generation with bank failed: {e}", exc_info=True)
            return None, None
    
    def _apply_feature_selection(
        self,
        features: np.ndarray,
        feature_names: List[str],
        selected_feature_names: List[str]
    ) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """
        Apply feature selection by selecting only the specified feature names.
        
        Args:
            features: Feature matrix
            feature_names: List of all feature names
            selected_feature_names: List of selected feature names from training
            
        Returns:
            Tuple of (selected_features_array, selected_feature_names_list)
        """
        try:
            if not selected_feature_names:
                return features, feature_names
            
            # Find indices of selected features
            selected_indices = []
            selected_names = []
            
            for name in selected_feature_names:
                if name in feature_names:
                    idx = feature_names.index(name)
                    selected_indices.append(idx)
                    selected_names.append(name)
                else:
                    self.logger.warning(f"Selected feature '{name}' not found in generated features")
            
            if not selected_indices:
                self.logger.warning("No selected features found in generated features")
                return features, feature_names
            
            # Select features
            selected_features = features[:, selected_indices]
            
            return selected_features, selected_names
            
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            return None, None
    
    def load_selected_features_from_artifacts(
        self,
        artifacts_path: Union[str, Path]
    ) -> bool:
        """
        Load selected feature names from training artifacts.
        
        Args:
            artifacts_path: Path to training artifacts file
            
        Returns:
            True if loaded successfully
        """
        try:
            artifacts_path = Path(artifacts_path)
            if not artifacts_path.exists():
                self.logger.warning(f"Artifacts file not found: {artifacts_path}")
                return False
            
            # Try loading from pickle file
            if artifacts_path.suffix == '.pkl':
                with open(artifacts_path, 'rb') as f:
                    artifacts = pickle.load(f)
                
                # Extract selected feature names from various possible structures
                if isinstance(artifacts, dict):
                    # Check component_result
                    if 'component_result' in artifacts:
                        result = artifacts['component_result']
                        if isinstance(result, dict):
                            # Check for feature_selection_info
                            if 'feature_selection_info' in result:
                                fs_info = result['feature_selection_info']
                                if isinstance(fs_info, dict):
                                    self.selected_feature_names = fs_info.get('selected_feature_names', [])
                                    self.feature_selection_info = fs_info
                            
                            # Check for selected_feature_names directly
                            if 'selected_feature_names' in result and not self.selected_feature_names:
                                self.selected_feature_names = result['selected_feature_names']
                    
                    # Check training_result
                    if 'training_result' in artifacts:
                        result = artifacts['training_result']
                        if isinstance(result, dict):
                            if 'feature_selection' in result:
                                fs_info = result['feature_selection']
                                if isinstance(fs_info, dict):
                                    self.selected_feature_names = fs_info.get('selected_feature_names', [])
                                    self.feature_selection_info = fs_info
                    
                    # Check top-level
                    if 'feature_selection_info' in artifacts:
                        fs_info = artifacts['feature_selection_info']
                        if isinstance(fs_info, dict):
                            self.selected_feature_names = fs_info.get('selected_feature_names', [])
                            self.feature_selection_info = fs_info
                    
                    if 'selected_feature_names' in artifacts and not self.selected_feature_names:
                        self.selected_feature_names = artifacts['selected_feature_names']
            
            # Try loading from JSON metadata
            metadata_path = artifacts_path.parent / "regime_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if 'selected_feature_names' in metadata and not self.selected_feature_names:
                        self.selected_feature_names = metadata['selected_feature_names']
            
            if self.selected_feature_names:
                self.logger.info(f"Loaded {len(self.selected_feature_names)} selected feature names")
                return True
            else:
                self.logger.warning("No selected feature names found in artifacts")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load selected features: {e}")
            return False


def create_regime_feature_engineer(
    selected_feature_names: Optional[List[str]] = None,
    feature_selection_info: Optional[Dict[str, Any]] = None,
    artifacts_path: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None
) -> RegimeFeatureEngineer:
    """
    Factory function to create RegimeFeatureEngineer.
    
    Args:
        selected_feature_names: List of selected feature names
        feature_selection_info: Feature selection metadata
        artifacts_path: Path to training artifacts to load selected features
        logger: Optional logger
        
    Returns:
        Initialized RegimeFeatureEngineer
    """
    engineer = RegimeFeatureEngineer(
        selected_feature_names=selected_feature_names,
        feature_selection_info=feature_selection_info,
        logger=logger
    )
    
    # Load from artifacts if provided
    if artifacts_path:
        engineer.load_selected_features_from_artifacts(artifacts_path)
    
    return engineer
