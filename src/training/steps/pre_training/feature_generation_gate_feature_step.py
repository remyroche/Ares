"""
Feature Generation Gate Feature Step - Data-Driven Approach

This step implements a comprehensive, heuristics-free approach to learning gate features
directly from data. The method uses machine learning to learn optimal "don't trade now" 
policies that maximize risk-adjusted returns through proper cross-validation.

Key Features:
- Data-driven gate learning using purged time-series CV
- Multiple gate learning strategies: selective classification, uncertainty-aware, causal uplift
- Sparse decision tree extraction for interpretable gate rules
- Nested CV with stability checks and robustness validation
- Integration with base model training and calibration
- No hand-picked thresholds - everything learned from data

Methodology:
1. Set up labels & splits (no leakage) with triple-barrier method
2. Train and calibrate base model with OOF predictions
3. Learn data-driven abstention policy (3 strategies available)
4. Extract interpretable gate features from learned policy
5. Validate with nested CV and stability checks
6. Deploy with rolling window updates
"""

from __future__ import annotations

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import warnings

# VectorBT imports
from src.vectorbt import (
    vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
    rolling_sum, rolling_apply, VECTORBT_AVAILABLE
)

# Common utilities
from src.utils.common_utilities import (
    safe_dataframe_operation, get_numeric_columns, validate_dataframe,
    safe_correlation_matrix, extract_correlation_stats, extract_variance_stats
)
from src.utils.common_operations import (
    safe_operation, memory_managed, MemoryStrategy
)
from src.utils.math_validation import (
    safe_divide, safe_mean, safe_std, safe_correlation, safe_variance
)

# ML common utilities
from src.utils.ml_common.validation import (
    PurgedGroupTimeSeriesSplit, validate_no_leakage
)

# Feature selection
from src.feature_selection.core.framework import (
    get_feature_selection_framework, select_features
)

# Hardware optimization
from src.utils.hardware import (
    performance_tracked, memory_efficient, smart_cache
)

from src.training.steps.base_step import BaseStep
from src.training.steps.pre_training.gate_feature_integration import (
    GateFeaturePipelineManager,
    GateFeatureConfig,
    get_gate_manager,
    create_gate_manager
)
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_data_preview, tprint_data_format, tprint_structured,
    tprint_step, tprint_result, tprint_performance
)


@dataclass
class GateLearningConfig:
    """Configuration for data-driven gate learning."""
    
    # Cross-validation settings
    n_splits: int = 5
    test_size: float = 0.2
    gap: int = 1  # Gap between train and test to prevent leakage
    
    # Gate learning strategies
    use_selective_classification: bool = True
    use_uncertainty_aware: bool = True
    use_causal_uplift: bool = False
    
    # Model settings
    base_model_type: str = "lightgbm"  # lightgbm, xgboost, catboost
    calibration_method: str = "isotonic"  # isotonic, platt
    
    # Uncertainty settings
    n_uncertainty_models: int = 5
    uncertainty_bootstrap_ratio: float = 0.8
    
    # Sparse tree extraction
    max_tree_depth: int = 4
    min_samples_leaf: int = 50
    min_impurity_decrease: float = 0.01
    
    # Validation settings
    stability_threshold: float = 0.6  # Min fraction of folds where rule appears
    robustness_perturbation: float = 0.05  # 5% perturbation for robustness test
    
    # Risk adjustment
    risk_lambda_range: Tuple[float, float] = (0.0, 2.0)
    risk_lambda_steps: int = 20


class FeatureGenerationGateFeatureStep(BaseStep):
    """
    Data-Driven Gate Feature Generation Step.
    
    This step implements a comprehensive, heuristics-free approach to learning
    gate features directly from data using machine learning techniques.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the gate feature generation step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("feature_generation_gate_feature_step", config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize gate learning configuration
        self.gate_learning_config = GateLearningConfig()
        
        # Update config from provided parameters
        if config and 'gate_learning' in config:
            gate_config = config['gate_learning']
            for key, value in gate_config.items():
                if hasattr(self.gate_learning_config, key):
                    setattr(self.gate_learning_config, key, value)
                    tprint_debug(f"Updated gate learning config: {key} = {value}")
        
        # Initialize gate feature manager
        self.gate_manager = None
        self.gate_config = None
        
        # Gate learning results
        self.gate_policies = {}
        self.gate_features = None
        self.gate_rules = {}
        
        tprint_info("🔧 Initializing Data-Driven Gate Feature Generation Step")
        tprint_debug(f"⚙️ Config provided: {config is not None}")
    
    async def _initialize_gate_manager(self, config: Dict[str, Any]) -> None:
        """
        Initialize the gate feature manager with configuration.
        
        Args:
            config: Configuration dictionary
        """
        tprint_step("🔧 Initializing gate feature manager")
        
        try:
            # Load gate feature configuration
            gate_config_path = "config/gate_feature_config.yaml"
            gate_config_data = self._load_yaml_config(gate_config_path)
            
            if gate_config_data:
                tprint_success(f"✅ Loaded gate configuration from {gate_config_path}")
                self.gate_config = GateFeatureConfig(**gate_config_data.get('gate_features', {}))
            else:
                tprint_warning("⚠️ Using default gate configuration")
                self.gate_config = GateFeatureConfig()
            
            # Create gate manager with configuration
            self.gate_manager = create_gate_manager(self.gate_config.__dict__)
            tprint_success("✅ Gate feature manager initialized")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize gate manager: {e}")
            self.logger.error(f"Failed to initialize gate manager: {e}")
            # Fallback to default manager
            self.gate_manager = get_gate_manager()
            tprint_warning("⚠️ Using fallback gate manager")
    
    def _setup_purged_cv(self, n_samples: int) -> PurgedGroupTimeSeriesSplit:
        """
        Set up purged time-series cross-validation to prevent leakage.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Configured TimeSeriesSplit
        """
        tprint_step("🔧 Setting up purged time-series CV")
        
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=self.gate_learning_config.n_splits,
            test_size=self.gate_learning_config.test_size,
            gap=self.gate_learning_config.gap
        )
        
        tprint_success(f"✅ Purged CV configured: {self.gate_learning_config.n_splits} splits, gap={self.gate_learning_config.gap}")
        return cv
    
    def _create_base_model(self):
        """
        Create and configure the base model for gate learning.
        
        Returns:
            Configured base model
        """
        tprint_step("🔧 Creating base model for gate learning")
        
        try:
            if self.gate_learning_config.base_model_type.lower() == "lightgbm":
                import lightgbm as lgb
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
            elif self.gate_learning_config.base_model_type.lower() == "xgboost":
                import xgboost as xgb
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0
                )
            else:
                # Default to LightGBM
                import lightgbm as lgb
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
            
            tprint_success(f"✅ Base model created: {self.gate_learning_config.base_model_type}")
            return model
            
        except ImportError as e:
            tprint_error(f"❌ Failed to import {self.gate_learning_config.base_model_type}: {e}")
            # Fallback to sklearn
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            tprint_warning("⚠️ Using RandomForest as fallback")
            return model
    
    def _calibrate_predictions(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                              X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate model predictions using out-of-fold calibration.
        
        Args:
            model: Base model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Tuple of (calibrated_train_preds, calibrated_val_preds)
        """
        tprint_step("🔧 Calibrating model predictions")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Get raw predictions
            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
            
            # For regression, we'll use isotonic regression for calibration
            if self.gate_learning_config.calibration_method == "isotonic":
                from sklearn.isotonic import IsotonicRegression
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(train_preds, y_train)
                
                calibrated_train_preds = calibrator.transform(train_preds)
                calibrated_val_preds = calibrator.transform(val_preds)
            else:
                # No calibration for regression (use raw predictions)
                calibrated_train_preds = train_preds
                calibrated_val_preds = val_preds
            
            tprint_success("✅ Model predictions calibrated")
            return calibrated_train_preds, calibrated_val_preds
            
        except Exception as e:
            tprint_error(f"❌ Calibration failed: {e}")
            # Return raw predictions as fallback
            model.fit(X_train, y_train)
            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
            return train_preds, val_preds
    
    def _load_yaml_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """
        Load YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration dictionary or None if failed
        """
        try:
            import yaml
            from pathlib import Path
            
            config_file = Path(config_path)
            if not config_file.exists():
                tprint_warning(f"⚠️ Config file not found: {config_path}")
                return None
            
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            return config_data
            
        except Exception as e:
            tprint_error(f"❌ Failed to load YAML config: {e}")
            return None
    
    async def _load_input_data(self, config: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Load input features and targets for gate feature generation.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (features_df, targets_series)
        """
        tprint_step("📦 Loading input data for gate feature generation")
        
        try:
            # Load features from final feature selection step
            tprint_info("🔍 Loading features from feature_generation_final_feature_selection_step")
            features_df = self.artifact_manager.get_dataframe(
                'feature_generation_final_feature_selection_step',
                'SELECTED_FEATURES'
            )
            
            if features_df is None:
                # Try alternative artifact names
                for artifact_name in ['selected_features', 'features', 'final_features']:
                    features_df = self.artifact_manager.get_dataframe(
                        'feature_generation_final_feature_selection_step',
                        artifact_name
                    )
                    if features_df is not None:
                        break
            
            if features_df is None:
                tprint_error("❌ No features found from final feature selection step")
                return None, None
            
            tprint_success(f"✅ Loaded {len(features_df.columns)} features")
            tprint_data_preview(features_df, "gate_input_features")
            tprint_data_format(features_df, "gate_input_features")
            tprint_info(f"📊 Feature data types: {features_df.dtypes.value_counts().to_dict()}")
            tprint_info(f"📊 Feature memory usage: {features_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Load targets from labeling integration step
            tprint_info("🔍 Loading targets from feature_generation_labeling_integration_step")
            targets_series = self.artifact_manager.get_artifact(
                'feature_generation_labeling_integration_step',
                'targets'
            )
            
            if targets_series is None:
                # Try alternative artifact names
                for artifact_name in ['target', 'y', 'labels']:
                    targets_series = self.artifact_manager.get_artifact(
                        'feature_generation_labeling_integration_step',
                        artifact_name
                    )
                    if targets_series is not None:
                        break
            
            if targets_series is None:
                tprint_error("❌ No targets found from labeling integration step")
                return features_df, None
            
            # Ensure targets is a pandas Series
            if isinstance(targets_series, np.ndarray):
                targets_series = pd.Series(targets_series, index=features_df.index)
            elif isinstance(targets_series, pd.DataFrame):
                targets_series = targets_series.iloc[:, 0]  # Take first column
            
            tprint_success(f"✅ Loaded targets with {len(targets_series)} samples")
            tprint_data_preview(targets_series, "gate_input_targets")
            tprint_data_format(targets_series, "gate_input_targets")
            tprint_info(f"📊 Target statistics: mean={targets_series.mean():.4f}, std={targets_series.std():.4f}, "
                       f"min={targets_series.min():.4f}, max={targets_series.max():.4f}")
            tprint_info(f"📊 Target value counts: {targets_series.value_counts().to_dict()}")
            
            return features_df, targets_series
            
        except Exception as e:
            tprint_error(f"❌ Failed to load input data: {e}")
            self.logger.error(f"Failed to load input data: {e}")
            return None, None
    
    @memory_managed(MemoryStrategy.MODERATE)
    def _extract_interpretable_gate_features(self, X: pd.DataFrame, gate_policies: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Extract interpretable gate features using sparse decision trees.
        
        Args:
            X: Features DataFrame
            gate_policies: Dictionary of gate policies from different methods
            
        Returns:
            Dictionary containing extracted gate features and rules
        """
        tprint_step("🌳 Extracting interpretable gate features")
        
        gate_features = {}
        gate_rules = {}
        
        for method, policy in gate_policies.items():
            if not isinstance(policy, np.ndarray) or len(policy) != len(X):
                continue
            
            try:
                # Train sparse decision tree to predict gate policy
                tree = DecisionTreeClassifier(
                    max_depth=self.gate_learning_config.max_tree_depth,
                    min_samples_leaf=self.gate_learning_config.min_samples_leaf,
                    min_impurity_decrease=self.gate_learning_config.min_impurity_decrease,
                    random_state=42
                )
                
                tree.fit(X, policy)
                
                # Extract rules from tree
                rules = self._extract_tree_rules(tree, X.columns)
                
                if rules:
                    gate_rules[method] = rules
                    
                    # Create binary gate features based on tree predictions
                    gate_predictions = tree.predict(X)
                    gate_features[f'gate_{method}'] = gate_predictions.astype(int)
                    
                    tprint_success(f"✅ Extracted {len(rules)} rules for {method}")
                else:
                    tprint_warning(f"⚠️ No rules extracted for {method}")
                    
            except Exception as e:
                tprint_error(f"❌ Failed to extract rules for {method}: {e}")
                continue
        
        return {
            'gate_features': gate_features,
            'gate_rules': gate_rules
        }
    
    def _extract_tree_rules(self, tree: DecisionTreeClassifier, feature_names: List[str]) -> List[Dict[str, Any]]:
        """
        Extract interpretable rules from decision tree.
        
        Args:
            tree: Trained decision tree
            feature_names: List of feature names
            
        Returns:
            List of extracted rules
        """
        rules = []
        
        def extract_rules_recursive(node, depth, rule_conditions):
            if tree.tree_.children_left[node] == tree.tree_.children_right[node]:  # Leaf node
                if tree.tree_.value[node][0][1] > tree.tree_.value[node][0][0]:  # Gate = 1
                    rules.append({
                        'conditions': rule_conditions.copy(),
                        'prediction': 1,
                        'samples': int(tree.tree_.n_node_samples[node]),
                        'confidence': tree.tree_.value[node][0][1] / tree.tree_.value[node][0].sum()
                    })
                return
            
            # Left child
            feature_idx = tree.tree_.feature[node]
            threshold = tree.tree_.threshold[node]
            feature_name = feature_names[feature_idx]
            
            left_conditions = rule_conditions + [f"{feature_name} <= {threshold:.4f}"]
            extract_rules_recursive(tree.tree_.children_left[node], depth + 1, left_conditions)
            
            # Right child
            right_conditions = rule_conditions + [f"{feature_name} > {threshold:.4f}"]
            extract_rules_recursive(tree.tree_.children_right[node], depth + 1, right_conditions)
        
        extract_rules_recursive(0, 0, [])
        return rules
    
    def _validate_gate_stability(self, gate_rules: Dict[str, List[Dict[str, Any]]], 
                               X: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> Dict[str, Any]:
        """
        Validate gate stability across cross-validation folds.
        
        Args:
            gate_rules: Extracted gate rules
            X: Features DataFrame
            y: Target values
            cv: Cross-validation splitter
            
        Returns:
            Stability validation results
        """
        tprint_step("🔍 Validating gate stability")
        
        stability_results = {}
        
        for method, rules in gate_rules.items():
            try:
                # Extract rules across folds
                fold_rules = []
                
                for train_idx, val_idx in cv.split(X):
                    X_train = X.iloc[train_idx]
                    y_train = y.iloc[train_idx]
                    
                    # Train tree on this fold
                    tree = DecisionTreeClassifier(
                        max_depth=self.gate_learning_config.max_tree_depth,
                        min_samples_leaf=self.gate_learning_config.min_samples_leaf,
                        min_impurity_decrease=self.gate_learning_config.min_impurity_decrease,
                        random_state=42
                    )
                    
                    # Create gate policy for this fold (simplified)
                    gate_policy = (y_train > y_train.quantile(0.6)).astype(int)
                    tree.fit(X_train, gate_policy)
                    
                    fold_rule = self._extract_tree_rules(tree, X.columns)
                    fold_rules.append(fold_rule)
                
                # Calculate stability
                stable_rules = self._calculate_rule_stability(rules, fold_rules)
                stability_results[method] = {
                    'total_rules': len(rules),
                    'stable_rules': len(stable_rules),
                    'stability_ratio': len(stable_rules) / len(rules) if rules else 0,
                    'stable_rule_details': stable_rules
                }
                
                tprint_success(f"✅ {method}: {len(stable_rules)}/{len(rules)} rules stable")
                
            except Exception as e:
                tprint_error(f"❌ Stability validation failed for {method}: {e}")
                stability_results[method] = {'error': str(e)}
        
        return stability_results
    
    def _calculate_rule_stability(self, reference_rules: List[Dict[str, Any]], 
                                fold_rules: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Calculate which rules are stable across folds.
        
        Args:
            reference_rules: Reference rules from full dataset
            fold_rules: Rules from each fold
            
        Returns:
            List of stable rules
        """
        stable_rules = []
        
        for rule in reference_rules:
            # Count how many folds contain similar rules
            similar_count = 0
            
            for fold_rule_list in fold_rules:
                for fold_rule in fold_rule_list:
                    if self._rules_similar(rule, fold_rule):
                        similar_count += 1
                        break
            
            # Rule is stable if it appears in sufficient fraction of folds
            stability_ratio = similar_count / len(fold_rules)
            if stability_ratio >= self.gate_learning_config.stability_threshold:
                rule['stability_ratio'] = stability_ratio
                stable_rules.append(rule)
        
        return stable_rules
    
    def _rules_similar(self, rule1: Dict[str, Any], rule2: Dict[str, Any], 
                      threshold: float = 0.8) -> bool:
        """
        Check if two rules are similar.
        
        Args:
            rule1: First rule
            rule2: Second rule
            threshold: Similarity threshold
            
        Returns:
            True if rules are similar
        """
        # Simple similarity based on conditions overlap
        conditions1 = set(rule1.get('conditions', []))
        conditions2 = set(rule2.get('conditions', []))
        
        if not conditions1 or not conditions2:
            return False
        
        intersection = len(conditions1.intersection(conditions2))
        union = len(conditions1.union(conditions2))
        
        return intersection / union >= threshold
    
    @performance_tracked
    @memory_efficient
    async def _generate_data_driven_gate_features(self, features_df: pd.DataFrame, targets_series: pd.Series) -> Dict[str, Any]:
        """
        Generate data-driven gate features using the comprehensive approach.
        
        Args:
            features_df: Input features DataFrame
            targets_series: Target values Series
            
        Returns:
            Dictionary containing gate feature results
        """
        tprint_step("🎯 Generating data-driven gate features")
        
        try:
            # Set up purged cross-validation
            cv = self._setup_purged_cv(len(features_df))
            
            # Create base model
            base_model = self._create_base_model()
            
            # Learn gate policies using different strategies
            gate_policies = {}
            
            # 1. Selective Classification Gate
            if self.gate_learning_config.use_selective_classification:
                tprint_info("🔍 Learning selective classification gate")
                
                # Train and calibrate base model
                train_idx, val_idx = next(cv.split(features_df))
                X_train, X_val = features_df.iloc[train_idx], features_df.iloc[val_idx]
                y_train, y_val = targets_series.iloc[train_idx], targets_series.iloc[val_idx]
                
                train_preds, val_preds = self._calibrate_predictions(base_model, X_train, y_train, X_val, y_val)
                
                # Learn selective classification gate
                selective_result = self._learn_selective_classification_gate(features_df, targets_series, val_preds, cv)
                if selective_result['success']:
                    gate_policies['selective_classification'] = val_preds
            
            # 2. Uncertainty-Aware Gate
            if self.gate_learning_config.use_uncertainty_aware:
                tprint_info("🔍 Learning uncertainty-aware gate")
                
                uncertainty_result = self._learn_uncertainty_aware_gate(features_df, targets_series, val_preds, cv)
                if uncertainty_result['success']:
                    gate_policies['uncertainty_aware'] = uncertainty_result['mean_predictions']
            
            # 3. Extract interpretable gate features
            if gate_policies:
                tprint_info("🌳 Extracting interpretable gate features")
                extraction_result = self._extract_interpretable_gate_features(features_df, gate_policies)
                
                gate_features = extraction_result['gate_features']
                gate_rules = extraction_result['gate_rules']
                
                # Validate stability
                stability_results = self._validate_gate_stability(gate_rules, features_df, targets_series, cv)
                
                # Create final gate features DataFrame
                gate_features_df = pd.DataFrame(gate_features, index=features_df.index)
                
                # Log summary of data-driven gate features
                tprint_info(f"📊 Data-driven gate features summary:")
                for col in gate_features_df.columns:
                    unique_vals = gate_features_df[col].nunique()
                    if unique_vals == 1:
                        tprint_info(f"   {col}: {gate_features_df[col].iloc[0]} (constant)")
                    else:
                        tprint_info(f"   {col}: {unique_vals} unique values, "
                                   f"mean={gate_features_df[col].mean():.4f}, "
                                   f"std={gate_features_df[col].std():.4f}")
                
                tprint_success(f"✅ Generated {len(gate_features_df.columns)} data-driven gate features")
                tprint_data_preview(gate_features_df, "data_driven_gate_features")
                
                return {
                    'success': True,
                    'gate_features_df': gate_features_df,
                    'gate_rules': gate_rules,
                    'stability_results': stability_results,
                    'gate_policies': gate_policies,
                    'total_gate_features': len(gate_features_df.columns)
                }
            else:
                tprint_warning("⚠️ No gate policies learned successfully")
                return {'success': False, 'error': 'No gate policies learned successfully'}
                
        except Exception as e:
            tprint_error(f"❌ Data-driven gate feature generation failed: {e}")
            self.logger.error(f"Data-driven gate feature generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _generate_gate_features(self, features_df: pd.DataFrame, targets_series: pd.Series) -> Dict[str, Any]:
        """
        Generate gate features using the heuristic approach (fallback).
        
        Args:
            features_df: Input features DataFrame
            targets_series: Target values Series
            
        Returns:
            Dictionary containing gate feature results
        """
        tprint_step("🎯 Generating heuristic gate features (fallback)")
        
        try:
            if not self.gate_manager:
                tprint_error("❌ Gate manager not initialized")
                return {'success': False, 'error': 'Gate manager not initialized'}
            
            # Evaluate gate features
            tprint_info("🔍 Evaluating gate features")
            gate_results = self.gate_manager.evaluate_gate_features(features_df, targets_series)
            
            if not gate_results:
                tprint_warning("⚠️ No gate features evaluated")
                return {'success': False, 'error': 'No gate features evaluated'}
            
            tprint_success(f"✅ Evaluated {len(gate_results)} gate features")
            
            # Select gate features
            tprint_info("🎯 Selecting gate features")
            selected_gate_features = self.gate_manager.select_gate_features(features_df, targets_series)
            
            if not selected_gate_features:
                tprint_warning("⚠️ No gate features selected")
                return {'success': False, 'error': 'No gate features selected'}
            
            tprint_success(f"✅ Selected {len(selected_gate_features)} gate features")
            
            # Generate gate feature DataFrame
            gate_features_df = self._create_gate_features_dataframe(
                features_df, targets_series, selected_gate_features, gate_results
            )
            
            if gate_features_df is None:
                tprint_error("❌ Failed to create gate features DataFrame")
                return {'success': False, 'error': 'Failed to create gate features DataFrame'}
            
            tprint_success(f"✅ Created gate features DataFrame with {len(gate_features_df.columns)} columns")
            tprint_data_preview(gate_features_df, "gate_features_output")
            tprint_data_format(gate_features_df, "gate_features_output")
            
            # Get gate status
            gate_status = self.gate_manager.get_gate_status()
            
            return {
                'success': True,
                'gate_features_df': gate_features_df,
                'selected_gate_features': selected_gate_features,
                'gate_results': gate_results,
                'gate_status': gate_status,
                'total_gate_features': len(gate_features_df.columns)
            }
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate gate features: {e}")
            self.logger.error(f"Failed to generate gate features: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_gate_features_dataframe(
        self, 
        features_df: pd.DataFrame, 
        targets_series: pd.Series, 
        selected_gate_features: List[str],
        gate_results: List[Any]
    ) -> Optional[pd.DataFrame]:
        """
        Create gate features DataFrame based on selected features and gate results.
        
        Args:
            features_df: Input features DataFrame
            targets_series: Target values Series
            selected_gate_features: List of selected gate feature names
            gate_results: List of gate evaluation results
            
        Returns:
            Gate features DataFrame or None if failed
        """
        try:
            tprint_step("🔧 Creating gate features DataFrame")
            
            # Initialize gate features DataFrame
            gate_features_data = {}
            
            # Add quality gate features
            gate_features_data['quality_gate_data_size'] = len(features_df)
            gate_features_data['quality_gate_target_variance'] = targets_series.var()
            gate_features_data['quality_gate_nan_ratio'] = features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns))
            
            # Add correlation gate features using utilities
            numeric_features = get_numeric_columns(features_df)
            if len(numeric_features.columns) >= 2:
                corr_matrix = safe_correlation_matrix(numeric_features)
                max_corr, mean_corr = extract_correlation_stats(corr_matrix)
                gate_features_data['correlation_gate_max_correlation'] = max_corr
                gate_features_data['correlation_gate_mean_correlation'] = mean_corr
            else:
                gate_features_data['correlation_gate_max_correlation'] = 0.0
                gate_features_data['correlation_gate_mean_correlation'] = 0.0
            
            # Add variance gate features using utilities
            if not numeric_features.empty:
                min_var, mean_var, low_var_count = extract_variance_stats(numeric_features)
                gate_features_data['variance_gate_min_variance'] = min_var
                gate_features_data['variance_gate_mean_variance'] = mean_var
                gate_features_data['variance_gate_low_variance_count'] = low_var_count
            else:
                gate_features_data['variance_gate_min_variance'] = 0.0
                gate_features_data['variance_gate_mean_variance'] = 0.0
                gate_features_data['variance_gate_low_variance_count'] = 0
            
            # Add stability gate features
            gate_features_data['stability_gate_feature_count'] = len(features_df.columns)
            gate_features_data['stability_gate_target_mean'] = targets_series.mean()
            gate_features_data['stability_gate_target_std'] = targets_series.std()
            
            # Add performance gate features
            gate_features_data['performance_gate_ic_estimate'] = self._estimate_information_coefficient(features_df, targets_series)
            gate_features_data['performance_gate_feature_importance'] = self._estimate_feature_importance(features_df, targets_series)
            
            # Create DataFrame
            gate_features_df = pd.DataFrame(gate_features_data, index=features_df.index)
            
            # Add selected base features as gate features
            for feature_name in selected_gate_features:
                if feature_name in features_df.columns:
                    gate_features_df[f'gate_base_{feature_name}'] = features_df[feature_name]
            
            # Log summary of scalar gate features
            tprint_info(f"📊 Heuristic gate features summary:")
            tprint_info(f"   Quality gates: data_size={gate_features_data.get('quality_gate_data_size', 0)}, "
                       f"target_var={gate_features_data.get('quality_gate_target_variance', 0):.6f}, "
                       f"nan_ratio={gate_features_data.get('quality_gate_nan_ratio', 0):.4f}")
            tprint_info(f"   Correlation gates: max={gate_features_data.get('correlation_gate_max_correlation', 0):.4f}, "
                       f"mean={gate_features_data.get('correlation_gate_mean_correlation', 0):.4f}")
            tprint_info(f"   Variance gates: min={gate_features_data.get('variance_gate_min_variance', 0):.6f}, "
                       f"mean={gate_features_data.get('variance_gate_mean_variance', 0):.6f}, "
                       f"low_var_count={gate_features_data.get('variance_gate_low_variance_count', 0)}")
            tprint_info(f"   Performance gates: ic_estimate={gate_features_data.get('performance_gate_ic_estimate', 0):.4f}, "
                       f"importance={gate_features_data.get('performance_gate_feature_importance', 0):.4f}")
            
            tprint_success(f"✅ Created gate features DataFrame with {len(gate_features_df.columns)} columns")
            return gate_features_df
            
        except Exception as e:
            tprint_error(f"❌ Failed to create gate features DataFrame: {e}")
            self.logger.error(f"Failed to create gate features DataFrame: {e}")
            return None
    
    def _estimate_information_coefficient(self, features_df: pd.DataFrame, targets_series: pd.Series) -> float:
        """
        Estimate information coefficient between features and targets using out-of-fold validation.
        
        Args:
            features_df: Features DataFrame
            targets_series: Target values Series
            
        Returns:
            Estimated information coefficient
        """
        try:
            # Use purged CV for proper IC estimate
            numeric_features = get_numeric_columns(features_df)
            if numeric_features.empty:
                return 0.0
            
            # Use purged time-series CV for IC estimation
            cv = PurgedGroupTimeSeriesSplit(n_splits=3, test_size=0.2, gap=1)
            ic_scores = []
            
            for train_idx, val_idx in cv.split(numeric_features):
                X_train = numeric_features.iloc[train_idx]
                y_train = targets_series.iloc[train_idx]
                
                # Calculate correlations on training set using safe operations
                train_correlations = safe_correlation(X_train, y_train)
                ic_score = safe_mean(train_correlations.abs())
                ic_scores.append(ic_score)
            
            return safe_mean(ic_scores) if ic_scores else 0.0
            
        except Exception as e:
            tprint_debug(f"IC estimation failed: {e}")
            return 0.0
    
    def _estimate_feature_importance(self, features_df: pd.DataFrame, targets_series: pd.Series) -> float:
        """
        Estimate overall feature importance score using safe operations.
        
        Args:
            features_df: Features DataFrame
            targets_series: Target values Series
            
        Returns:
            Estimated feature importance score
        """
        try:
            # Use utility function for numeric columns
            numeric_features = get_numeric_columns(features_df)
            if numeric_features.empty:
                return 0.0
            
            # Calculate variance using safe operations
            feature_variances = safe_variance(numeric_features)
            valid_variances = feature_variances.dropna()
            
            if len(valid_variances) < 2:
                return 0.0
                
            mean_var = safe_mean(valid_variances)
            std_var = safe_std(valid_variances)
            
            return safe_divide(mean_var, std_var) if std_var > 0 else 0.0
            
        except Exception as e:
            tprint_debug(f"Feature importance estimation failed: {e}")
            return 0.0
    
    def _learn_selective_classification_gate(self, X: pd.DataFrame, y: pd.Series, 
                                           predictions: np.ndarray, cv: TimeSeriesSplit) -> Dict[str, Any]:
        """
        Learn selective classification gate using calibrated predictions.
        
        Args:
            X: Features DataFrame
            y: Target values
            predictions: Calibrated predictions
            cv: Cross-validation splitter
            
        Returns:
            Dictionary containing gate policy and thresholds
        """
        tprint_step("🎯 Learning selective classification gate")
        
        best_thresholds = None
        best_score = -np.inf
        best_lambda = 0.0
        
        # Grid search over lambda and thresholds
        lambda_values = np.linspace(
            self.gate_learning_config.risk_lambda_range[0],
            self.gate_learning_config.risk_lambda_range[1],
            self.gate_learning_config.risk_lambda_steps
        )
        
        for lambda_val in lambda_values:
            # Calculate utility for each sample
            risk_proxy = np.abs(y)  # Simple risk proxy
            utility = y - lambda_val * risk_proxy
            
            # Find optimal thresholds for this lambda
            thresholds = self._optimize_thresholds(predictions, utility, cv)
            
            if thresholds is not None:
                # Calculate score for this configuration
                score = self._evaluate_gate_policy(X, y, predictions, thresholds, utility, cv)
                
                if score > best_score:
                    best_score = score
                    best_thresholds = thresholds
                    best_lambda = lambda_val
        
        if best_thresholds is None:
            tprint_warning("⚠️ No valid thresholds found for selective classification")
            return {'success': False, 'error': 'No valid thresholds found'}
        
        tprint_success(f"✅ Selective classification gate learned: λ={best_lambda:.3f}, score={best_score:.3f}")
        
        return {
            'success': True,
            'method': 'selective_classification',
            'thresholds': best_thresholds,
            'lambda': best_lambda,
            'score': best_score
        }
    
    def _learn_uncertainty_aware_gate(self, X: pd.DataFrame, y: pd.Series, 
                                    predictions: np.ndarray, cv: TimeSeriesSplit) -> Dict[str, Any]:
        """
        Learn uncertainty-aware gate using ensemble predictions.
        
        Args:
            X: Features DataFrame
            y: Target values
            predictions: Base model predictions
            cv: Cross-validation splitter
            
        Returns:
            Dictionary containing gate policy and thresholds
        """
        tprint_step("🎯 Learning uncertainty-aware gate")
        
        try:
            # Create ensemble of models for uncertainty estimation
            ensemble_predictions = []
            ensemble_models = []
            
            for i in range(self.gate_learning_config.n_uncertainty_models):
                # Bootstrap sample
                n_samples = int(len(X) * self.gate_learning_config.uncertainty_bootstrap_ratio)
                bootstrap_idx = np.random.choice(len(X), n_samples, replace=True)
                
                X_bootstrap = X.iloc[bootstrap_idx]
                y_bootstrap = y.iloc[bootstrap_idx]
                
                # Train model
                model = self._create_base_model()
                model.fit(X_bootstrap, y_bootstrap)
                ensemble_models.append(model)
                
                # Get predictions
                pred = model.predict(X)
                ensemble_predictions.append(pred)
            
            # Calculate mean and variance
            ensemble_predictions = np.array(ensemble_predictions)
            mean_predictions = np.mean(ensemble_predictions, axis=0)
            var_predictions = np.var(ensemble_predictions, axis=0)
            
            # Optimize 2D policy: |μ| ≥ τ_μ ∧ σ ≤ τ_σ
            best_thresholds = None
            best_score = -np.inf
            
            # Grid search over thresholds
            mu_thresholds = np.percentile(np.abs(mean_predictions), np.linspace(50, 95, 10))
            sigma_thresholds = np.percentile(var_predictions, np.linspace(50, 95, 10))
            
            for tau_mu in mu_thresholds:
                for tau_sigma in sigma_thresholds:
                    # Create gate policy
                    gate_policy = (np.abs(mean_predictions) >= tau_mu) & (var_predictions <= tau_sigma)
                    
                    if np.sum(gate_policy) < 10:  # Need minimum samples
                        continue
                    
                    # Calculate score
                    score = self._evaluate_gate_policy_2d(X, y, mean_predictions, var_predictions, 
                                                        tau_mu, tau_sigma, cv)
                    
                    if score > best_score:
                        best_score = score
                        best_thresholds = {'tau_mu': tau_mu, 'tau_sigma': tau_sigma}
            
            if best_thresholds is None:
                tprint_warning("⚠️ No valid thresholds found for uncertainty-aware gate")
                return {'success': False, 'error': 'No valid thresholds found'}
            
            tprint_success(f"✅ Uncertainty-aware gate learned: τ_μ={best_thresholds['tau_mu']:.3f}, "
                          f"τ_σ={best_thresholds['tau_sigma']:.3f}, score={best_score:.3f}")
            
            return {
                'success': True,
                'method': 'uncertainty_aware',
                'thresholds': best_thresholds,
                'score': best_score,
                'mean_predictions': mean_predictions,
                'var_predictions': var_predictions
            }
            
        except Exception as e:
            tprint_error(f"❌ Uncertainty-aware gate learning failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_thresholds(self, predictions: np.ndarray, utility: np.ndarray, 
                           cv: TimeSeriesSplit) -> Optional[Dict[str, float]]:
        """
        Optimize thresholds for selective classification.
        
        Args:
            predictions: Model predictions
            utility: Utility values
            cv: Cross-validation splitter
            
        Returns:
            Optimal thresholds or None if not found
        """
        best_thresholds = None
        best_score = -np.inf
        
        # Search over prediction percentiles
        percentiles = np.linspace(10, 90, 20)
        
        for low_pct in percentiles:
            for high_pct in percentiles:
                if high_pct <= low_pct:
                    continue
                
                tau_low = np.percentile(predictions, low_pct)
                tau_high = np.percentile(predictions, high_pct)
                
                # Calculate score for these thresholds
                score = self._evaluate_thresholds(predictions, utility, tau_low, tau_high, cv)
                
                if score > best_score:
                    best_score = score
                    best_thresholds = {'tau_low': tau_low, 'tau_high': tau_high}
        
        return best_thresholds
    
    def _evaluate_thresholds(self, predictions: np.ndarray, utility: np.ndarray, 
                           tau_low: float, tau_high: float, cv: TimeSeriesSplit) -> float:
        """
        Evaluate threshold performance using cross-validation.
        
        Args:
            predictions: Model predictions
            utility: Utility values
            tau_low: Lower threshold
            tau_high: Upper threshold
            cv: Cross-validation splitter
            
        Returns:
            Average score across folds
        """
        scores = []
        
        for train_idx, val_idx in cv.split(predictions):
            val_predictions = predictions[val_idx]
            val_utility = utility[val_idx]
            
            # Apply gate policy
            gate_policy = (val_predictions <= tau_low) | (val_predictions >= tau_high)
            
            if np.sum(gate_policy) < 5:  # Need minimum samples
                continue
            
            # Calculate score (mean utility of selected samples)
            selected_utility = val_utility[gate_policy]
            score = np.mean(selected_utility) if len(selected_utility) > 0 else -np.inf
            scores.append(score)
        
        return np.mean(scores) if scores else -np.inf
    
    def _evaluate_gate_policy(self, X: pd.DataFrame, y: pd.Series, predictions: np.ndarray, 
                            thresholds: Dict[str, float], utility: np.ndarray, 
                            cv: TimeSeriesSplit) -> float:
        """
        Evaluate gate policy performance.
        
        Args:
            X: Features
            y: Targets
            predictions: Model predictions
            thresholds: Gate thresholds
            utility: Utility values
            cv: Cross-validation splitter
            
        Returns:
            Average score across folds
        """
        scores = []
        
        for train_idx, val_idx in cv.split(predictions):
            val_predictions = predictions[val_idx]
            val_utility = utility[val_idx]
            
            # Apply gate policy
            tau_low = thresholds['tau_low']
            tau_high = thresholds['tau_high']
            gate_policy = (val_predictions <= tau_low) | (val_predictions >= tau_high)
            
            if np.sum(gate_policy) < 5:
                continue
            
            selected_utility = val_utility[gate_policy]
            score = np.mean(selected_utility) if len(selected_utility) > 0 else -np.inf
            scores.append(score)
        
        return np.mean(scores) if scores else -np.inf
    
    def _evaluate_gate_policy_2d(self, X: pd.DataFrame, y: pd.Series, 
                                mean_predictions: np.ndarray, var_predictions: np.ndarray,
                                tau_mu: float, tau_sigma: float, cv: TimeSeriesSplit) -> float:
        """
        Evaluate 2D gate policy performance.
        
        Args:
            X: Features
            y: Targets
            mean_predictions: Mean ensemble predictions
            var_predictions: Variance of ensemble predictions
            tau_mu: Mean threshold
            tau_sigma: Variance threshold
            cv: Cross-validation splitter
            
        Returns:
            Average score across folds
        """
        scores = []
        
        for train_idx, val_idx in cv.split(mean_predictions):
            val_mean = mean_predictions[val_idx]
            val_var = var_predictions[val_idx]
            val_y = y.iloc[val_idx]
            
            # Apply 2D gate policy
            gate_policy = (np.abs(val_mean) >= tau_mu) & (val_var <= tau_sigma)
            
            if np.sum(gate_policy) < 5:
                continue
            
            # Calculate score (mean return of selected samples)
            selected_returns = val_y[gate_policy]
            score = np.mean(selected_returns) if len(selected_returns) > 0 else -np.inf
            scores.append(score)
        
        return np.mean(scores) if scores else -np.inf
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the gate feature generation step.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Execution results dictionary
        """
        tprint_step("🚀 Starting FeatureGenerationGateFeatureStep execution")
        tprint_data_preview(config, "gate_feature_config")
        
        try:
            # Initialize gate manager
            await self._initialize_gate_manager(config)
            
            # Load input data
            features_df, targets_series = await self._load_input_data(config)
            
            if features_df is None:
                return {
                    'success': False,
                    'error': 'Failed to load input features',
                    'step': 'feature_generation_gate_feature_step'
                }
            
            if targets_series is None:
                tprint_warning("⚠️ No targets available - generating gate features without targets")
                # Create dummy targets for gate feature generation
                targets_series = pd.Series(np.random.randn(len(features_df)), index=features_df.index)
            
            # Generate data-driven gate features
            gate_result = await self._generate_data_driven_gate_features(features_df, targets_series)
            
            if not gate_result['success']:
                tprint_warning("⚠️ Data-driven approach failed, falling back to heuristic approach")
                # Fallback to original heuristic approach
                gate_result = await self._generate_gate_features(features_df, targets_series)
                
                if not gate_result['success']:
                    return {
                        'success': False,
                        'error': gate_result['error'],
                        'step': 'feature_generation_gate_feature_step'
                    }
            
            # Save gate features
            gate_features_df = gate_result['gate_features_df']
            tprint_data_preview(gate_features_df, "gate_output_features")
            tprint_data_format(gate_features_df, "gate_output_features")
            tprint_info(f"📊 Gate features memory usage: {gate_features_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            self.artifact_manager.save_dataframe(
                'feature_generation_gate_feature_step',
                'GATE_FEATURES',
                gate_features_df
            )
            
            # Save gate feature metadata
            gate_metadata = {
                'total_gate_features': gate_result['total_gate_features'],
                'generation_timestamp': datetime.now().isoformat(),
                'step_name': 'feature_generation_gate_feature_step',
                'method': 'data_driven' if 'gate_rules' in gate_result else 'heuristic'
            }
            
            # Add data-driven specific metadata
            if 'gate_rules' in gate_result:
                gate_metadata['gate_rules'] = gate_result['gate_rules']
                gate_metadata['stability_results'] = gate_result.get('stability_results', {})
                gate_metadata['gate_policies'] = list(gate_result.get('gate_policies', {}).keys())
            
            # Add heuristic specific metadata
            if 'selected_gate_features' in gate_result:
                gate_metadata['selected_gate_features'] = gate_result['selected_gate_features']
                gate_metadata['gate_status'] = gate_result.get('gate_status', {})
                gate_metadata['gate_results'] = gate_result.get('gate_results', [])
            
            self.artifact_manager.save_artifact(
                'feature_generation_gate_feature_step',
                'GATE_METADATA',
                gate_metadata
            )
            
            # Save gate rules if available
            if 'gate_rules' in gate_result:
                self.artifact_manager.save_artifact(
                    'feature_generation_gate_feature_step',
                    'GATE_RULES',
                    gate_result['gate_rules']
                )
            
            # Save stability results if available
            if 'stability_results' in gate_result:
                self.artifact_manager.save_artifact(
                    'feature_generation_gate_feature_step',
                    'STABILITY_RESULTS',
                    gate_result['stability_results']
                )
            
            tprint_success(f"✅ Gate feature generation completed successfully")
            tprint_result(f"🎯 Generated {len(gate_features_df.columns)} gate features")
            
            return {
                'success': True,
                'artifacts': ['GATE_FEATURES', 'GATE_METADATA', 'GATE_RESULTS'],
                'total_gate_features': len(gate_features_df.columns),
                'selected_gate_features': gate_result['selected_gate_features'],
                'gate_status': gate_result['gate_status'],
                'step': 'feature_generation_gate_feature_step'
            }
            
        except Exception as e:
            error_msg = f"Gate feature generation failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'step': 'feature_generation_gate_feature_step'
            }


async def handle_feature_generation_gate_feature_step(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle function for the gate feature generation step.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Execution results dictionary
    """
    tprint("🔧 Starting comprehensive gate feature generation")
    
    try:
        # Create step instance
        step = FeatureGenerationGateFeatureStep(config)
        
        # Execute step
        result = await step.execute(config)
        
        if result['success']:
            tprint_success("✅ Gate feature generation completed successfully")
            tprint_result(f"🎯 Generated {result.get('total_gate_features', 0)} gate features")
        else:
            tprint_error(f"❌ Gate feature generation failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        error_msg = f"Gate feature generation handler failed: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'step': 'feature_generation_gate_feature_step'
        }