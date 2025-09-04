#!/usr/bin/env python3
"""Multi-Output Model Trainer for Direction and Profit Prediction.

This module provides intelligent multi-output prediction capabilities for both
price direction and expected profit using the triple barrier method and
profit-based feature engineering.
"""
from src.core.decorators import (
    cached,
    handles_errors,
    log_execution_time,
    validate_dataframe,
)
from copy import copy

from src.core.domain import (
    comprehensive_validation,
    secure_data_processing
)

import json
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pathlib import Path
import asyncio

# Optional imports for additional model types
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

# Import existing model architectures from step06
try:
    from .steps.step9_hmm_based_training import (
        CNNModel, CNNTrainer,
        TCNModel, TCNTrainer,
        TransformerModel, TransformerTrainer
    )
    EXISTING_MODELS_AVAILABLE = True
except ImportError:
    EXISTING_MODELS_AVAILABLE = False
    CNNModel = CNNTrainer = TCNModel = TCNTrainer = TransformerModel = TransformerTrainer = None

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from src.training.steps.step06_labeling_components.profit_based_feature_engineering import (
    ProfitBasedFeatureEngineering
)

from src.utils.logger import system_logger

class MultiOutputModelConfig:
    """Configuration for multi-output model training."""
    
    def __init__(
        self,
        model_type: str = "LightGBM",
        direction_target: str = "direction",
        profit_target: str = "expected_profit",
        use_profit_features: bool = True,
        profit_feature_columns: Optional[List[str]] = None,
        direction_threshold: float = 0.0,
        profit_scaling: str = "standard",  # "standard", "robust", "minmax"
        ensemble_method: str = "stacking",  # "stacking", "voting", "blending"
        validation_method: str = "time_series_cv",
        n_splits: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        use_enhanced_feature_selection: bool = True,  # NEW: Use enhanced feature selection
        supported_model_types: List[str] = None,  # NEW: Supported model types
        # NEW: Probability output configuration
        enable_probability_outputs: bool = True,
        probability_targets: Optional[List[str]] = None,
        probability_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_type = model_type
        self.direction_target = direction_target
        self.profit_target = profit_target
        self.use_profit_features = use_profit_features
        self.profit_feature_columns = profit_feature_columns or []
        self.direction_threshold = direction_threshold
        self.profit_scaling = profit_scaling
        self.ensemble_method = ensemble_method
        self.validation_method = validation_method
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.use_enhanced_feature_selection = use_enhanced_feature_selection
        
        # NEW: Probability output configuration
        self.enable_probability_outputs = enable_probability_outputs
        if probability_targets is None:
            self.probability_targets = [
                "triple_barrier_probability",
                "direction_probability", 
                "magnitude_probability",
                "barrier_avoidance_probability"
            ]
        else:
            self.probability_targets = probability_targets
        
        # Default probability configuration
        if probability_config is None:
            self.probability_config = {
                "profit_target": 0.02,
                "stop_loss": 0.01,
                "look_ahead_periods": 20,
                "magnitude_threshold_factor": 0.8,
                "adverse_threshold": 0.01,
                "avoidance_look_ahead": 10
            }
        else:
            self.probability_config = probability_config
        
        # Supported model types for multi-output training
        if supported_model_types is None:
            self.supported_model_types = [
                "LightGBM", "RandomForest", "XGBoost", "CatBoost", "NeuralNetwork"
            ]
            # Add existing models if available
            if EXISTING_MODELS_AVAILABLE:
                self.supported_model_types.extend(["CNN", "TCN", "Transformer"])
        else:
            self.supported_model_types = supported_model_types

class MultiOutputNeuralNetwork(nn.Module):
    """Neural network for multi-output prediction (direction + profit)."""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        direction_output_size: int = 1,
        profit_output_size: int = 1,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # Shared layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Direction prediction head (classification)
        self.direction_head = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(prev_size // 2, direction_output_size),
            nn.Sigmoid()  # For binary classification
        )
        
        # Profit prediction head (regression)
        self.profit_head = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(prev_size // 2, profit_output_size)
        )
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        direction_pred = self.direction_head(shared_features)
        profit_pred = self.profit_head(shared_features)
        return direction_pred, profit_pred

class MultiOutputModelTrainer:
    """Multi-output model trainer for direction and profit prediction with comprehensive SR features."""
    
    def __init__(self, config: MultiOutputModelConfig):
        self.config = config
        self.logger = system_logger.getChild("MultiOutputModelTrainer")
        
        # Initialize profit-based feature engineering
        self.profit_feature_engine = ProfitBasedFeatureEngineering(
            profit_column="potential_profit_pct",
            use_numba=True,
            memory_efficient=True
        )
        
        # NEW: SR Feature Integration
        self.step7_features = []  # Features from step07
        self.step2_5_sr_levels = {}  # SR levels from step2_5
        self.sr_feature_columns = []  # All SR feature column names
        self.comprehensive_sr_features = {}  # Combined SR features
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        
        # Training history
        self.training_history = {
            "direction_metrics": [],
            "profit_metrics": [],
            "combined_metrics": [],
            "feature_importance": {},
            "training_time": 0.0,
            "sr_feature_analysis": {}  # NEW: SR feature analysis
        }
        
        # NEW: Probability training components
        if self.config.enable_probability_outputs:
            from .multi_output_probability_trainer import ProbabilityTargetGenerator
            self.probability_target_generator = ProbabilityTargetGenerator(self.config.probability_config)
            self.logger.info("🔧 Probability target generator initialized")
        
        self.logger.info("🔧 Multi-output model trainer initialized with comprehensive SR feature integration")

    @handles_errors(
        exceptions=(ValueError, FileNotFoundError, json.JSONDecodeError),
        default_return=False,
        context="step7_features_loading"
    )
    async def load_step7_features(self, step7_output_path: str) -> bool:
        """
        Load comprehensive SR features from step07 enhanced matrix operations.
        
        Args:
            step7_output_path: Path to step07 output directory
            
        Returns:
            bool: True if features loaded successfully
        """
        try:
            self.logger.info(f"📊 Loading step07 SR features from: {step7_output_path}")
            
            # Load step07 matrix operations results
            step7_results_path = Path(step7_output_path) / "matrix_operations_results.json"
            if not step7_results_path.exists():
                self.logger.warning(f"⚠️ Step7 results not found at: {step7_results_path}")
                return False
            
            with open(step7_results_path, 'r') as f:
                step7_results = json.load(f)
            
            # Extract SR features from step07 results
            sr_analysis = step7_results.get("sr_analysis", {})
            sr_enhanced_analysis = step7_results.get("sr_enhanced_analysis", {})
            sr_optimization_analysis = step7_results.get("sr_optimization_analysis", {})
            
            # Collect all SR features
            self.step7_features = []
            
            # Basic SR features
            basic_sr_features = sr_analysis.get("sr_features", [])
            self.step7_features.extend(basic_sr_features)
            
            # Enhanced SR features
            enhanced_sr_features = sr_enhanced_analysis.get("enhanced_sr_features", [])
            self.step7_features.extend(enhanced_sr_features)
            
            # Optimization SR features
            optimization_sr_features = sr_optimization_analysis.get("optimization_features", [])
            self.step7_features.extend(optimization_sr_features)
            
            # Remove duplicates and sort
            self.step7_features = sorted(list(set(self.step7_features)))
            
            self.logger.info(f"✅ Loaded {len(self.step7_features)} SR features from step07")
            self.logger.info(f"   - Basic SR features: {len(basic_sr_features)}")
            self.logger.info(f"   - Enhanced SR features: {len(enhanced_sr_features)}")
            self.logger.info(f"   - Optimization SR features: {len(optimization_sr_features)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading step07 features: {e}")
            return False

    @handles_errors(
        exceptions=(ValueError, FileNotFoundError, json.JSONDecodeError),
        default_return=False,
        context="step2_5_sr_levels_loading"
    )
    async def load_step2_5_sr_levels(self, step2_5_output_path: str) -> bool:
        """
        Load SR levels from step2_5 SR optimization.
        
        Args:
            step2_5_output_path: Path to step2_5 output directory
            
        Returns:
            bool: True if SR levels loaded successfully
        """
        try:
            self.logger.info(f"📊 Loading step2_5 SR levels from: {step2_5_output_path}")
            
            # Load step2_5 SR optimization results
            step2_5_results_path = Path(step2_5_output_path) / "sr_optimization_results.json"
            if not step2_5_results_path.exists():
                self.logger.warning(f"⚠️ Step2_5 results not found at: {step2_5_results_path}")
                return False
            
            with open(step2_5_results_path, 'r') as f:
                step2_5_results = json.load(f)
            
            # Extract SR levels
            self.step2_5_sr_levels = step2_5_results.get("sr_levels_result", {})
            
            support_levels = self.step2_5_sr_levels.get("support_levels", [])
            resistance_levels = self.step2_5_sr_levels.get("resistance_levels", [])
            
            self.logger.info(f"✅ Loaded SR levels from step2_5:")
            self.logger.info(f"   - Support levels: {len(support_levels)}")
            self.logger.info(f"   - Resistance levels: {len(resistance_levels)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading step2_5 SR levels: {e}")
            return False

    def convert_sr_levels_to_features(self, current_price: float) -> dict[str, float]:
        """
        Convert SR levels from step2_5 to ML features.
        
        Args:
            current_price: Current market price
            
        Returns:
            dict: SR level features
        """
        try:
            features = {}
            
            # Support level features
            support_levels = self.step2_5_sr_levels.get("support_levels", [])
            features.update({
                "sr_support_level_count": len(support_levels),
                "sr_nearest_support_distance": self._calculate_nearest_distance(support_levels, current_price),
                "sr_support_level_strength_avg": np.mean([level.get("strength", 0.5) for level in support_levels]) if support_levels else 0.5,
                "sr_support_level_volume_avg": np.mean([level.get("volume", 0) for level in support_levels]) if support_levels else 0.0,
                "sr_support_level_age_avg": np.mean([level.get("age", 0) for level in support_levels]) if support_levels else 0.0,
                "sr_support_level_touches_avg": np.mean([level.get("touches", 0) for level in support_levels]) if support_levels else 0.0,
            })
            
            # Resistance level features
            resistance_levels = self.step2_5_sr_levels.get("resistance_levels", [])
            features.update({
                "sr_resistance_level_count": len(resistance_levels),
                "sr_nearest_resistance_distance": self._calculate_nearest_distance(resistance_levels, current_price),
                "sr_resistance_level_strength_avg": np.mean([level.get("strength", 0.5) for level in resistance_levels]) if resistance_levels else 0.5,
                "sr_resistance_level_volume_avg": np.mean([level.get("volume", 0) for level in resistance_levels]) if resistance_levels else 0.0,
                "sr_resistance_level_age_avg": np.mean([level.get("age", 0) for level in resistance_levels]) if resistance_levels else 0.0,
                "sr_resistance_level_touches_avg": np.mean([level.get("touches", 0) for level in resistance_levels]) if resistance_levels else 0.0,
            })
            
            # Combined level features
            all_levels = support_levels + resistance_levels
            if all_levels:
                price_range = max([level.get("price", current_price) for level in all_levels]) - min([level.get("price", current_price) for level in all_levels])
                price_range = max(price_range, current_price * 0.01)  # Minimum range
                
                features.update({
                    "sr_total_levels": len(all_levels),
                    "sr_level_density": len(all_levels) / price_range if price_range > 0 else 0.0,
                    "sr_level_strength_variance": np.var([level.get("strength", 0.5) for level in all_levels]),
                    "sr_level_volume_variance": np.var([level.get("volume", 0) for level in all_levels]),
                    "sr_level_age_variance": np.var([level.get("age", 0) for level in all_levels]),
                })
            else:
                features.update({
                    "sr_total_levels": 0,
                    "sr_level_density": 0.0,
                    "sr_level_strength_variance": 0.0,
                    "sr_level_volume_variance": 0.0,
                    "sr_level_age_variance": 0.0,
                })
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error converting SR levels to features: {e}")
            return self._get_default_sr_level_features()

    def _calculate_nearest_distance(self, levels: list, current_price: float) -> float:
        """Calculate distance to nearest level."""
        if not levels:
            return 1.0  # Far away if no levels
        
        distances = [abs(level.get("price", current_price) - current_price) / current_price for level in levels]
        return min(distances) if distances else 1.0

    def _get_default_sr_level_features(self) -> dict[str, float]:
        """Return default SR level features when conversion fails."""
        return {
            "sr_support_level_count": 0, "sr_nearest_support_distance": 1.0, "sr_support_level_strength_avg": 0.5,
            "sr_support_level_volume_avg": 0.0, "sr_support_level_age_avg": 0.0, "sr_support_level_touches_avg": 0.0,
            "sr_resistance_level_count": 0, "sr_nearest_resistance_distance": 1.0, "sr_resistance_level_strength_avg": 0.5,
            "sr_resistance_level_volume_avg": 0.0, "sr_resistance_level_age_avg": 0.0, "sr_resistance_level_touches_avg": 0.0,
            "sr_total_levels": 0, "sr_level_density": 0.0, "sr_level_strength_variance": 0.0,
            "sr_level_volume_variance": 0.0, "sr_level_age_variance": 0.0
        }

    def validate_feature_completeness(self, features_df: pd.DataFrame) -> dict[str, list[str]]:
        """
        Validate that all required SR features are present.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            dict: Missing features by category
        """
        try:
            required_features = {
                # Step7 SR features (42 features)
                "step7_sr_features": [
                    "sr_proximity", "support_proximity", "resistance_proximity", "sr_zone_width",
                    "sr_strength", "support_strength", "resistance_strength", "sr_enhanced_strength",
                    "sr_total_support_levels", "sr_total_resistance_levels", "sr_clusters_detected",
                    "sr_fibonacci_levels", "sr_elliott_waves", "sr_order_flow_imbalances",
                    "sr_distance", "normalized_distance", "sr_proximity_score", "sr_zone_position_pct",
                    "strength_score", "sr_enhanced_support_strength", "sr_enhanced_resistance_strength",
                    "sr_optimized_strength_weights", "sr_noise_points", "sr_clustering_quality",
                    "sr_level", "sr_order_flow_poc", "sr_order_flow_hvns", "sr_optimized_fibonacci_sensitivity",
                    "sr_optimized_elliott_confidence", "sr_optimized_order_flow_threshold",
                    "sr_touch_count", "sr_bounce_rate", "sr_isolation_score", "sr_momentum_pct",
                    "sr_volatility_pct", "sr_trend_pct", "sr_optimization_score", "sr_optimized_method_weights",
                    "sr_optimized_dbscan_eps", "sr_optimized_dbscan_min_samples", "delta_sr_score", "clarity_factor"
                ],
                
                # Step2_5 SR level features (15 features)
                "step2_5_sr_level_features": [
                    "sr_support_level_count", "sr_nearest_support_distance", "sr_support_level_strength_avg",
                    "sr_resistance_level_count", "sr_nearest_resistance_distance", "sr_resistance_level_strength_avg",
                    "sr_total_levels", "sr_level_density", "sr_level_strength_variance",
                    "sr_support_level_volume_avg", "sr_support_level_age_avg", "sr_support_level_touches_avg",
                    "sr_resistance_level_volume_avg", "sr_resistance_level_age_avg", "sr_resistance_level_touches_avg",
                    "sr_level_volume_variance", "sr_level_age_variance"
                ]
            }
            
            missing_features = {}
            for category, features in required_features.items():
                missing = [f for f in features if f not in features_df.columns]
                if missing:
                    missing_features[category] = missing
                    
            if missing_features:
                self.logger.warning(f"⚠️ Missing SR features: {missing_features}")
            else:
                self.logger.info("✅ All required SR features are present")
                
            return missing_features
            
        except Exception as e:
            self.logger.error(f"❌ Error validating feature completeness: {e}")
            return {}

    async def _add_comprehensive_sr_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive SR features from step07 and step2_5 to the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with comprehensive SR features added
        """
        try:
            self.logger.info("🔧 Adding comprehensive SR features...")
            
            # Create a copy to avoid modifying original data
            data_with_sr = data.copy()
            
            # Add step07 SR features if available
            if self.step7_features:
                self.logger.info(f"📊 Adding {len(self.step7_features)} step07 SR features...")
                
                # Initialize step07 features with default values
                for feature in self.step7_features:
                    if feature not in data_with_sr.columns:
                        data_with_sr[feature] = 0.5  # Default neutral value
                
                self.logger.info(f"✅ Added step07 SR features: {len(self.step7_features)} features")
            
            # Add step2_5 SR level features
            if self.step2_5_sr_levels:
                self.logger.info("📊 Adding step2_5 SR level features...")
                
                # Get current prices for SR level feature calculation
                if 'close' in data_with_sr.columns:
                    current_prices = data_with_sr['close'].values
                else:
                    # Use a default price if close column not available
                    current_prices = [100.0] * len(data_with_sr)
                
                # Calculate SR level features for each row
                sr_level_features_list = []
                for i, current_price in enumerate(current_prices):
                    sr_level_features = self.convert_sr_levels_to_features(current_price)
                    sr_level_features_list.append(sr_level_features)
                
                # Convert to DataFrame and add to main data
                sr_level_df = pd.DataFrame(sr_level_features_list, index=data_with_sr.index)
                data_with_sr = pd.concat([data_with_sr, sr_level_df], axis=1)
                
                self.logger.info(f"✅ Added step2_5 SR level features: {len(sr_level_df.columns)} features")
            
            # Create combined SR features
            data_with_sr = self._create_combined_sr_features(data_with_sr)
            
            # Validate feature completeness
            missing_features = self.validate_feature_completeness(data_with_sr)
            if missing_features:
                self.logger.warning(f"⚠️ Some SR features are missing: {missing_features}")
            
            # Store SR feature columns for later use
            self.sr_feature_columns = [col for col in data_with_sr.columns if 'sr_' in col.lower()]
            self.logger.info(f"📊 Total SR features available: {len(self.sr_feature_columns)}")
            
            return data_with_sr
            
        except Exception as e:
            self.logger.error(f"❌ Error adding comprehensive SR features: {e}")
            return data

    def _create_combined_sr_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create combined SR features from individual SR features.
        
        Args:
            data: DataFrame with individual SR features
            
        Returns:
            pd.DataFrame: DataFrame with combined SR features added
        """
        try:
            # Combined proximity features
            if 'sr_proximity' in data.columns and 'sr_zone_width' in data.columns:
                data['sr_proximity_zone_ratio'] = data['sr_proximity'] / (data['sr_zone_width'] + 1e-8)
            
            # Combined strength features
            if 'sr_strength' in data.columns and 'sr_enhanced_strength' in data.columns:
                data['sr_strength_enhanced_ratio'] = data['sr_strength'] / (data['sr_enhanced_strength'] + 1e-8)
            
            # Combined level features
            if 'sr_support_level_count' in data.columns and 'sr_resistance_level_count' in data.columns:
                data['sr_support_resistance_ratio'] = data['sr_support_level_count'] / (data['sr_resistance_level_count'] + 1e-8)
                data['sr_total_levels'] = data['sr_support_level_count'] + data['sr_resistance_level_count']
            
            # Combined distance features
            if 'sr_nearest_support_distance' in data.columns and 'sr_nearest_resistance_distance' in data.columns:
                data['sr_nearest_level_distance'] = np.minimum(
                    data['sr_nearest_support_distance'], 
                    data['sr_nearest_resistance_distance']
                )
                data['sr_distance_ratio'] = data['sr_nearest_support_distance'] / (data['sr_nearest_resistance_distance'] + 1e-8)
            
            # SR momentum features
            if 'sr_momentum_pct' in data.columns and 'sr_volatility_pct' in data.columns:
                data['sr_momentum_volatility_ratio'] = data['sr_momentum_pct'] / (data['sr_volatility_pct'] + 1e-8)
            
            # SR trend features
            if 'sr_trend_pct' in data.columns and 'sr_momentum_pct' in data.columns:
                data['sr_trend_momentum_alignment'] = np.sign(data['sr_trend_pct']) * np.sign(data['sr_momentum_pct'])
            
            self.logger.info("✅ Created combined SR features")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Error creating combined SR features: {e}")
            return data

    def _analyze_sr_features(self, features_df: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze SR features in the dataset.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            dict: SR feature analysis statistics
        """
        try:
            # Get SR feature columns
            sr_columns = [col for col in features_df.columns if 'sr_' in col.lower()]
            
            if not sr_columns:
                return {"sr_feature_count": 0, "sr_feature_categories": {}}
            
            # Analyze SR features by category
            categories = {
                "proximity": [col for col in sr_columns if any(keyword in col.lower() for keyword in ['proximity', 'distance', 'nearest'])],
                "strength": [col for col in sr_columns if any(keyword in col.lower() for keyword in ['strength', 'enhanced'])],
                "levels": [col for col in sr_columns if any(keyword in col.lower() for keyword in ['level', 'support', 'resistance'])],
                "momentum": [col for col in sr_columns if any(keyword in col.lower() for keyword in ['momentum', 'trend', 'volatility'])],
                "advanced": [col for col in sr_columns if any(keyword in col.lower() for keyword in ['fibonacci', 'elliott', 'order_flow', 'clustering'])],
                "optimization": [col for col in sr_columns if any(keyword in col.lower() for keyword in ['optimized', 'optimization'])],
                "combined": [col for col in sr_columns if any(keyword in col.lower() for keyword in ['ratio', 'alignment', 'combined'])]
            }
            
            # Calculate statistics for each category
            category_stats = {}
            for category, cols in categories.items():
                if cols:
                    category_stats[category] = {
                        "count": len(cols),
                        "features": cols,
                        "mean_values": features_df[cols].mean().to_dict(),
                        "std_values": features_df[cols].std().to_dict()
                    }
            
            # Overall statistics
            overall_stats = {
                "sr_feature_count": len(sr_columns),
                "sr_feature_categories": category_stats,
                "total_features": len(features_df.columns),
                "sr_feature_percentage": len(sr_columns) / len(features_df.columns) * 100
            }
            
            return overall_stats
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing SR features: {e}")
            return {"sr_feature_count": 0, "error": str(e)}
    
    @handles_errors(
        exceptions=(ValueError, TypeError, MemoryError),
        default_return=None,
        context="multi_output_data_preparation"
    )
    async def prepare_multi_output_data(
        self,
        data: pd.DataFrame,
        direction_column: str = "direction",
        profit_column: str = "potential_profit_pct",
        feature_columns: Optional[List[str]] = None,
        use_enhanced_feature_selection: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for multi-output training with comprehensive SR features."
        
        Args:
            data: Input DataFrame with features and targets
            direction_column: Column name for direction target
            profit_column: Column name for profit target
            feature_columns: List of feature columns to use
            use_enhanced_feature_selection: Whether to use enhanced feature selection with autoencoder features
            
        Returns:
            Tuple of (features, direction_target, profit_target)
        """
        self.logger.info("📊 Preparing multi-output training data with comprehensive SR features...")
        
        # Validate input data
        if data.empty:
            raise ValueError("Input data is empty")
        
        required_columns = [direction_column, profit_column]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # NEW: Add comprehensive SR features
        data_with_sr_features = await self._add_comprehensive_sr_features(data)
        
        # Use enhanced data-driven feature selection if enabled
        if use_enhanced_feature_selection:
            try:
                from src.training.steps.step09_hmm_based_training import HMMBasedTrainingStep
                
                self.logger.info("🔧 Using enhanced data-driven feature selection (VIF, MI, SHAP, RF)...")
                
                # Create step06 instance for feature selection
                step6_config = {"symbol": "default", "exchange": "default", "data_dir": "temp"}
                step6_instance = Step6HMMBasedTraining(step6_config)
                
                # Use the enhanced pre-filtering method
                selected_features = await step6_instance._pre_filter_features(
                    X=data_with_sr_features,
                    feature_columns=[col for col in data_with_sr_features.columns if col not in [direction_column, profit_column]]
                )
                
                # Add back target columns
                selected_features.extend([direction_column, profit_column])
                selected_features = [col for col in selected_features if col in data_with_sr_features.columns]
                
                self.logger.info(f"✅ Enhanced data-driven feature selection completed: {len(selected_features)} features selected")
                
                # Use selected features
                data = data[selected_features]
                
            except Exception as e:
                self.logger.warning(f"⚠️ Enhanced data-driven feature selection failed: {e}")
                self.logger.info("📊 Falling back to basic feature preparation")
                use_enhanced_feature_selection = False
        
        # Apply profit-based feature engineering if enabled and not using enhanced selection
        if self.config.use_profit_features and not use_enhanced_feature_selection:
            self.logger.info("🔧 Applying profit-based feature engineering...")
            data = self.profit_feature_engine.apply_all_features(data)
            self.logger.info(f"✅ Added profit-based features")
        
        # Select features
        if feature_columns is None:
            # Use all columns except targets and metadata
            exclude_columns = [
                direction_column, profit_column, "timestamp", "timeframe",
                "composite_cluster_id", "sample_weight"
            ]
            feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # Prepare features and targets
        features = data[feature_columns].copy()
        direction_target = data[direction_column].copy()
        profit_target = data[profit_column].copy()
        
        # Handle missing values
        features = features.fillna(0)
        direction_target = direction_target.fillna(0)
        profit_target = profit_target.fillna(0)
        
        # Convert direction to binary if needed
        if direction_target.dtype in ['object', 'string']:
            # Assume positive direction is 1, negative is 0
            direction_target = (direction_target > self.config.direction_threshold).astype(int)
        
        self.logger.info(f"✅ Prepared data: {features.shape[0]} samples, {features.shape[1]} features")
        self.logger.info(f"   - Direction target: {direction_target.value_counts().to_dict()}")
        self.logger.info(f"   - Profit target: mean={profit_target.mean():.6f}, std={profit_target.std():.6f}")
        
        return features, direction_target, profit_target
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get a summary of feature importance scores from data-driven selection."
        
        Returns:
            Dictionary containing feature importance summaries by method
        """
        if not hasattr(self, 'feature_importance') or not self.feature_importance:
            return {}
        
        summary = {}
        
        for method, scores in self.feature_importance.items():
            if isinstance(scores, dict) and len(scores) > 0:
                scores_series = pd.Series(scores)
                summary[method] = {
                    "mean": float(scores_series.mean()),
                    "std": float(scores_series.std()),
                    "min": float(scores_series.min()),
                    "max": float(scores_series.max()),
                    "top_10_features": scores_series.nlargest(10).to_dict()
                }
        
        return summary
    
    def _train_xgboost_multi_output(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_dir_train: np.ndarray,
        y_dir_val: np.ndarray,
        y_prof_train: np.ndarray,
        y_prof_val: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Train XGBoost multi-output model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Please install xgboost package.")
        
        self.logger.info("🌳 Training XGBoost multi-output model...")
        
        # Train direction classifier
        direction_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.config.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        direction_model.fit(
            X_train, y_dir_train,
            eval_set=[(X_val, y_dir_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Train profit regressor
        profit_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.config.random_state,
            eval_metric='rmse'
        )
        profit_model.fit(
            X_train, y_prof_train,
            eval_set=[(X_val, y_prof_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Evaluate models
        direction_pred = direction_model.predict(X_val)
        profit_pred = profit_model.predict(X_val)
        
        direction_accuracy = accuracy_score(y_dir_val, direction_pred)
        profit_rmse = np.sqrt(mean_squared_error(y_prof_val, profit_pred))
        
        # Calculate metrics
        direction_metrics = {
            "accuracy": direction_accuracy,
            "f1": f1_score(y_dir_val, direction_pred),
            "precision": precision_score(y_dir_val, direction_pred),
            "recall": recall_score(y_dir_val, direction_pred)
        }
        
        profit_metrics = {
            "rmse": profit_rmse,
            "mae": mean_absolute_error(y_prof_val, profit_pred),
            "r2": r2_score(y_prof_val, profit_pred)
        }
        
        combined_metrics = {
            "direction_accuracy": direction_accuracy,
            "profit_rmse": profit_rmse,
            "overall_score": direction_accuracy - profit_rmse  # Simple combination
        }
        
        return {
            "direction_model": direction_model,
            "profit_model": profit_model,
            "model_type": "XGBoost",
            "direction_metrics": direction_metrics,
            "profit_metrics": profit_metrics,
            "combined_metrics": combined_metrics,
            "feature_importance": {
                "direction": direction_model.feature_importances_,
                "profit": profit_model.feature_importances_
            }
        }
    
    def _train_catboost_multi_output(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_dir_train: np.ndarray,
        y_dir_val: np.ndarray,
        y_prof_train: np.ndarray,
        y_prof_val: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Train CatBoost multi-output model."""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not available. Please install catboost package.")
        
        self.logger.info("🐱 Training CatBoost multi-output model...")
        
        # Train direction classifier
        direction_model = cb.CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=self.config.random_state,
            verbose=False
        )
        direction_model.fit(
            X_train, y_dir_train,
            eval_set=(X_val, y_dir_val),
            early_stopping_rounds=10
        )
        
        # Train profit regressor
        profit_model = cb.CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=self.config.random_state,
            verbose=False
        )
        profit_model.fit(
            X_train, y_prof_train,
            eval_set=(X_val, y_prof_val),
            early_stopping_rounds=10
        )
        
        # Evaluate models
        direction_pred = direction_model.predict(X_val)
        profit_pred = profit_model.predict(X_val)
        
        direction_accuracy = accuracy_score(y_dir_val, direction_pred)
        profit_rmse = np.sqrt(mean_squared_error(y_prof_val, profit_pred))
        
        # Calculate metrics
        direction_metrics = {
            "accuracy": direction_accuracy,
            "f1": f1_score(y_dir_val, direction_pred),
            "precision": precision_score(y_dir_val, direction_pred),
            "recall": recall_score(y_dir_val, direction_pred)
        }
        
        profit_metrics = {
            "rmse": profit_rmse,
            "mae": mean_absolute_error(y_prof_val, profit_pred),
            "r2": r2_score(y_prof_val, profit_pred)
        }
        
        combined_metrics = {
            "direction_accuracy": direction_accuracy,
            "profit_rmse": profit_rmse,
            "overall_score": direction_accuracy - profit_rmse  # Simple combination
        }
        
        return {
            "direction_model": direction_model,
            "profit_model": profit_model,
            "model_type": "CatBoost",
            "direction_metrics": direction_metrics,
            "profit_metrics": profit_metrics,
            "combined_metrics": combined_metrics,
            "feature_importance": {
                "direction": direction_model.feature_importances_,
                "profit": profit_model.feature_importances_
            }
        }
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError),
        default_return=None,
        context="multi_output_model_training"
    )
    @log_execution_time
    @cached
    async def train_multi_output_model(
        self,
        features: pd.DataFrame,
        direction_target: pd.Series,
        profit_target: pd.Series,
        model_name: str = "multi_output_model"
    ) -> Dict[str, Any]:
        """Train a multi-output model for direction and profit prediction with comprehensive SR features."
        
        Args:
            features: Feature DataFrame
            direction_target: Direction target series
            profit_target: Profit target series
            model_name: Name for the trained model
            
        Returns:
            Dictionary containing training results and model artifacts
        """
        start_time = time.time()
        self.logger.info(f"🚀 Training multi-output model with comprehensive SR features: {model_name}")
        
        # NEW: Validate SR feature completeness
        missing_features = self.validate_feature_completeness(features)
        if missing_features:
            self.logger.warning(f"⚠️ Missing SR features: {missing_features}")
        
        # NEW: Log SR feature statistics
        sr_feature_stats = self._analyze_sr_features(features)
        self.logger.info(f"📊 SR Feature Statistics: {sr_feature_stats}")
        
        # Prepare data
        X = features.values
        y_direction = direction_target.values
        y_profit = profit_target.values
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        # Initialize results storage
        direction_metrics = []
        profit_metrics = []
        combined_metrics = []
        
        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"🔄 Training fold {fold + 1}/{self.config.n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_dir_train, y_dir_val = y_direction[train_idx], y_direction[val_idx]
            y_prof_train, y_prof_val = y_profit[train_idx], y_profit[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model based on type
            if self.config.model_type == "LightGBM":
                model_result = self._train_lightgbm_multi_output(
                    X_train_scaled, X_val_scaled,
                    y_dir_train, y_dir_val,
                    y_prof_train, y_prof_val,
                    features.columns
                )
            elif self.config.model_type == "RandomForest":
                model_result = self._train_randomforest_multi_output(
                    X_train_scaled, X_val_scaled,
                    y_dir_train, y_dir_val,
                    y_prof_train, y_prof_val,
                    features.columns
                )
            elif self.config.model_type == "XGBoost":
                model_result = self._train_xgboost_multi_output(
                    X_train_scaled, X_val_scaled,
                    y_dir_train, y_dir_val,
                    y_prof_train, y_prof_val,
                    features.columns
                )
            elif self.config.model_type == "CatBoost":
                model_result = self._train_catboost_multi_output(
                    X_train_scaled, X_val_scaled,
                    y_dir_train, y_dir_val,
                    y_prof_train, y_prof_val,
                    features.columns
                )
            elif self.config.model_type == "NeuralNetwork":
                model_result = self._train_neural_network_multi_output(
                    X_train_scaled, X_val_scaled,
                    y_dir_train, y_dir_val,
                    y_prof_train, y_prof_val,
                    features.columns
                )
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}. Supported types: {self.config.supported_model_types}")
            
            if model_result:
                direction_metrics.append(model_result["direction_metrics"])
                profit_metrics.append(model_result["profit_metrics"])
                combined_metrics.append(model_result["combined_metrics"])
        
        # Aggregate results
        if direction_metrics and profit_metrics:
            final_model = self._train_final_model(
                X, y_direction, y_profit, features.columns
            )
            
            # NEW: Store SR feature analysis in training history
            self.training_history["sr_feature_analysis"] = sr_feature_stats
            
            # Calculate average metrics
            avg_direction_metrics = self._aggregate_metrics(direction_metrics)
            avg_profit_metrics = self._aggregate_metrics(profit_metrics)
            avg_combined_metrics = self._aggregate_metrics(combined_metrics)
            
            # Store results
            training_time = time.time() - start_time
            result = {
                "model_name": model_name,
                "model": final_model,
                "scaler": scaler,
                "feature_columns": list(features.columns),
                "direction_metrics": avg_direction_metrics,
                "profit_metrics": avg_profit_metrics,
                "combined_metrics": avg_combined_metrics,
                "training_time": training_time,
                "config": self.config.__dict__
            }
            
            # Store model artifacts
            self.models[model_name] = final_model
            self.scalers[model_name] = scaler
            
            self.logger.info(f"✅ Multi-output model training completed in {training_time:.2f}s")
            self.logger.info(f"   - Direction accuracy: {avg_direction_metrics['accuracy']:.4f}")
            self.logger.info(f"   - Profit R²: {avg_profit_metrics['r2']:.4f}")
            
            return result
        else:
            self.logger.error("❌ No successful training results")
            return None
    
    def _train_lightgbm_multi_output(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_dir_train: np.ndarray,
        y_dir_val: np.ndarray,
        y_prof_train: np.ndarray,
        y_prof_val: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Train LightGBM multi-output model."""
        
        # Direction model (classification)
        direction_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.config.random_state,
            verbose=-1
        )
        
        direction_model.fit(
            X_train, y_dir_train,
            eval_set=[(X_val, y_dir_val)],
            eval_metric="binary_logloss",
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Profit model (regression)
        profit_model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.config.random_state,
            verbose=-1
        )
        
        profit_model.fit(
            X_train, y_prof_train,
            eval_set=[(X_val, y_prof_val)],
            eval_metric="rmse",
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Predictions
        y_dir_pred = direction_model.predict(X_val)
        y_prof_pred = profit_model.predict(X_val)
        
        # Metrics
        direction_metrics = {
            "accuracy": accuracy_score(y_dir_val, y_dir_pred),
            "precision": precision_score(y_dir_val, y_dir_pred, zero_division=0),
            "recall": recall_score(y_dir_val, y_dir_pred, zero_division=0),
            "f1": f1_score(y_dir_val, y_dir_pred, zero_division=0)
        }
        
        profit_metrics = {
            "mse": mean_squared_error(y_prof_val, y_prof_pred),
            "mae": mean_absolute_error(y_prof_val, y_prof_pred),
            "r2": r2_score(y_prof_val, y_prof_pred),
            "rmse": np.sqrt(mean_squared_error(y_prof_val, y_prof_pred))
        }
        
        # Combined metrics (direction-weighted profit)
        direction_profit = y_dir_pred * y_prof_pred
        actual_direction_profit = y_dir_val * y_prof_val
        
        combined_metrics = {
            "direction_weighted_profit_correlation": np.corrcoef(
                direction_profit, actual_direction_profit
            )[0, 1],
            "profit_accuracy": np.mean(np.sign(direction_profit) == np.sign(actual_direction_profit)),
            "total_profit_pred": np.sum(direction_profit),
            "total_profit_actual": np.sum(actual_direction_profit)
        }
        
        return {
            "direction_model": direction_model,
            "profit_model": profit_model,
            "direction_metrics": direction_metrics,
            "profit_metrics": profit_metrics,
            "combined_metrics": combined_metrics,
            "feature_importance": {
                "direction": dict(zip(feature_names, direction_model.feature_importances_)),
                "profit": dict(zip(feature_names, profit_model.feature_importances_))
            }
        }
    
    def _train_randomforest_multi_output(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_dir_train: np.ndarray,
        y_dir_val: np.ndarray,
        y_prof_train: np.ndarray,
        y_prof_val: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Train Random Forest multi-output model."""
        
        # Direction model (classification)
        direction_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        direction_model.fit(X_train, y_dir_train)
        
        # Profit model (regression)
        profit_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        profit_model.fit(X_train, y_prof_train)
        
        # Predictions
        y_dir_pred = direction_model.predict(X_val)
        y_prof_pred = profit_model.predict(X_val)
        
        # Metrics (same as LightGBM)
        direction_metrics = {
            "accuracy": accuracy_score(y_dir_val, y_dir_pred),
            "precision": precision_score(y_dir_val, y_dir_pred, zero_division=0),
            "recall": recall_score(y_dir_val, y_dir_pred, zero_division=0),
            "f1": f1_score(y_dir_val, y_dir_pred, zero_division=0)
        }
        
        profit_metrics = {
            "mse": mean_squared_error(y_prof_val, y_prof_pred),
            "mae": mean_absolute_error(y_prof_val, y_prof_pred),
            "r2": r2_score(y_prof_val, y_prof_pred),
            "rmse": np.sqrt(mean_squared_error(y_prof_val, y_prof_pred))
        }
        
        # Combined metrics
        direction_profit = y_dir_pred * y_prof_pred
        actual_direction_profit = y_dir_val * y_prof_val
        
        combined_metrics = {
            "direction_weighted_profit_correlation": np.corrcoef(
                direction_profit, actual_direction_profit
            )[0, 1],
            "profit_accuracy": np.mean(np.sign(direction_profit) == np.sign(actual_direction_profit)),
            "total_profit_pred": np.sum(direction_profit),
            "total_profit_actual": np.sum(actual_direction_profit)
        }
        
        return {
            "direction_model": direction_model,
            "profit_model": profit_model,
            "direction_metrics": direction_metrics,
            "profit_metrics": profit_metrics,
            "combined_metrics": combined_metrics,
            "feature_importance": {
                "direction": dict(zip(feature_names, direction_model.feature_importances_)),
                "profit": dict(zip(feature_names, profit_model.feature_importances_))
            }
        }
    
    def _train_neural_network_multi_output(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_dir_train: np.ndarray,
        y_dir_val: np.ndarray,
        y_prof_train: np.ndarray,
        y_prof_val: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Train Neural Network multi-output model."""
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_dir_train_tensor = torch.FloatTensor(y_dir_train).unsqueeze(1)
        y_dir_val_tensor = torch.FloatTensor(y_dir_val).unsqueeze(1)
        y_prof_train_tensor = torch.FloatTensor(y_prof_train).unsqueeze(1)
        y_prof_val_tensor = torch.FloatTensor(y_prof_val).unsqueeze(1)
        
        # Create model
        model = MultiOutputNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=[128, 64, 32],
            dropout_rate=0.2
        )
        
        # Training setup
        criterion_direction = nn.BCELoss()
        criterion_profit = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(50):  # Simplified training
            optimizer.zero_grad()
            
            dir_pred, prof_pred = model(X_train_tensor)
            
            loss_direction = criterion_direction(dir_pred, y_dir_train_tensor)
            loss_profit = criterion_profit(prof_pred, y_prof_train_tensor)
            
            total_loss = loss_direction + loss_profit
            total_loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            dir_pred_val, prof_pred_val = model(X_val_tensor)
            
            y_dir_pred = (dir_pred_val.squeeze() > 0.5).float().numpy()
            y_prof_pred = prof_pred_val.squeeze().numpy()
        
        # Metrics (same as other models)
        direction_metrics = {
            "accuracy": accuracy_score(y_dir_val, y_dir_pred),
            "precision": precision_score(y_dir_val, y_dir_pred, zero_division=0),
            "recall": recall_score(y_dir_val, y_dir_pred, zero_division=0),
            "f1": f1_score(y_dir_val, y_dir_pred, zero_division=0)
        }
        
        profit_metrics = {
            "mse": mean_squared_error(y_prof_val, y_prof_pred),
            "mae": mean_absolute_error(y_prof_val, y_prof_pred),
            "r2": r2_score(y_prof_val, y_prof_pred),
            "rmse": np.sqrt(mean_squared_error(y_prof_val, y_prof_pred))
        }
        
        # Combined metrics
        direction_profit = y_dir_pred * y_prof_pred
        actual_direction_profit = y_dir_val * y_prof_val
        
        combined_metrics = {
            "direction_weighted_profit_correlation": np.corrcoef(
                direction_profit, actual_direction_profit
            )[0, 1],
            "profit_accuracy": np.mean(np.sign(direction_profit) == np.sign(actual_direction_profit)),
            "total_profit_pred": np.sum(direction_profit),
            "total_profit_actual": np.sum(actual_direction_profit)
        }
        
        return {
            "direction_model": model,
            "profit_model": model,  # Same model for both outputs
            "direction_metrics": direction_metrics,
            "profit_metrics": profit_metrics,
            "combined_metrics": combined_metrics,
            "feature_importance": {
                "direction": {},  # Neural networks don't have direct feature importance
                "profit": {}
            }
        }
    
    def _train_final_model(
        self,
        X: np.ndarray,
        y_direction: np.ndarray,
        y_profit: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Train final model on full dataset."""
        
        if self.config.model_type == "LightGBM":
            direction_model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.config.random_state,
                verbose=-1
            )
            
            profit_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.config.random_state,
                verbose=-1
            )
            
            direction_model.fit(X, y_direction)
            profit_model.fit(X, y_profit)
            
            return {
                "direction_model": direction_model,
                "profit_model": profit_model,
                "model_type": "LightGBM"
            }
        
        elif self.config.model_type == "RandomForest":
            direction_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
            profit_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
            direction_model.fit(X, y_direction)
            profit_model.fit(X, y_profit)
            
            return {
                "direction_model": direction_model,
                "profit_model": profit_model,
                "model_type": "RandomForest"
            }
        
        else:
            raise ValueError(f"Unsupported model type for final training: {self.config.model_type}")
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across folds."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [metrics[key] for metrics in metrics_list if key in metrics]
            if values:
                aggregated[key] = np.mean(values)
        
        return aggregated
    
    def predict(
        self,
        features: pd.DataFrame,
        model_name: str = "multi_output_model",
        current_prices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions using trained multi-output model."
        
        Args:
            features: Feature DataFrame
            model_name: Name of the model to use
            current_prices: Current price levels for price prediction (optional)
            
        Returns:
            Tuple of (direction_predictions, profit_predictions, price_predictions)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        # Scale features
        X_scaled = scaler.transform(features.values)
        
        # Make predictions
        if model["model_type"] in ["LightGBM", "RandomForest"]:
            direction_pred = model["direction_model"].predict(X_scaled)
            profit_pred = model["profit_model"].predict(X_scaled)
            
            # Calculate price predictions if current prices are provided
            if current_prices is not None:
                # Price prediction = current_price * (1 + profit_prediction)
                price_pred = current_prices * (1 + profit_pred)
            else:
                # If no current prices, return profit predictions as price changes
                price_pred = profit_pred
        else:
            raise ValueError(f"Unsupported model type for prediction: {model['model_type']}")
        
        return direction_pred, profit_pred, price_pred
    
    def predict_with_confidence(
        self,
        features: pd.DataFrame,
        model_name: str = "multi_output_model",
        current_prices: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.7
    ) -> Dict[str, np.ndarray]:
        """Make predictions with confidence scoring using existing confidence utility."
        
        Args:
            features: Feature DataFrame
            model_name: Name of the model to use
            current_prices: Current price levels for price prediction (optional)
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        from src.utils.confidence import calculate_multi_output_confidence_batch, get_confidence_threshold_signals
        
        # Make basic predictions
        direction_pred, profit_pred, price_pred = self.predict(
            features, model_name, current_prices
        )
        
        # Get direction probabilities (for confidence calculation)
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        X_scaled = scaler.transform(features.values)
        
        if model["model_type"] in ["LightGBM", "RandomForest"]:
            direction_prob = model["direction_model"].predict_proba(X_scaled)[:, 1]
        else:
            # Fallback: use prediction as probability
            direction_prob = direction_pred.astype(float)
        
        # Use current prices or default to ones
        if current_prices is None:
            current_prices = np.ones_like(profit_pred)
        
        # Calculate confidence using existing utility
        confidence_scores = calculate_multi_output_confidence_batch(
            direction_probabilities=direction_prob,
            direction_predictions=direction_pred,
            profit_predictions=profit_pred,
            current_prices=current_prices,
            predicted_prices=price_pred,
            direction_threshold=0.6,
            profit_threshold=0.001,
            price_threshold=0.005,
            min_ensemble_confidence=confidence_threshold
        )
        
        # Get trading signals based on confidence threshold
        trading_signals = get_confidence_threshold_signals(
            confidence_scores, threshold=confidence_threshold
        )
        
        return {
            'direction_prediction': direction_pred,
            'profit_prediction': profit_pred,
            'price_prediction': price_pred,
            'direction_probability': direction_prob,
            'confidence_scores': confidence_scores,
            'trading_signals': trading_signals,
            'final_confidence': confidence_scores['final_confidence']
        }
    
    def save_model(
        self,
        model_name: str,
        save_path: str
    ) -> None:
        """Save trained model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        os.makedirs(save_path, exist_ok=True)
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        # Save model components
        if model["model_type"] in ["LightGBM", "RandomForest"]:
            joblib.dump(model["direction_model"], f"{save_path}/direction_model.pkl")
            joblib.dump(model["profit_model"], f"{save_path}/profit_model.pkl")
        
        # Save scaler
        joblib.dump(scaler, f"{save_path}/scaler.pkl")
        
        # Save metadata
        metadata = {
            "model_type": model["model_type"],
            "feature_columns": self.training_history.get("feature_columns", []),
            "config": self.config.__dict__,
            "training_history": self.training_history
        }
        
        with open(f"{save_path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"✅ Model saved to {save_path}")
    
    def load_model(
        self,
        model_name: str,
        load_path: str
    ) -> None:
        """Load trained model from disk."""
        # Load model components
        if os.path.exists(f"{load_path}/direction_model.pkl"):
            direction_model = joblib.load(f"{load_path}/direction_model.pkl")
            profit_model = joblib.load(f"{load_path}/profit_model.pkl")
            scaler = joblib.load(f"{load_path}/scaler.pkl")
            
            # Determine model type
            if hasattr(direction_model, 'feature_importances_'):
                model_type = "RandomForest" if hasattr(direction_model, 'estimators_') else "LightGBM"
            else:
                model_type = "Unknown"
            
            # Store loaded model
            self.models[model_name] = {
                "direction_model": direction_model,
                "profit_model": profit_model,
                "model_type": model_type
            }
            self.scalers[model_name] = scaler
            
            # Load metadata
            if os.path.exists(f"{load_path}/metadata.json"):
                with open(f"{load_path}/metadata.json", "r") as f:
                    metadata = json.load(f)
                    self.training_history = metadata.get("training_history", {})
            
            self.logger.info(f"✅ Model loaded from {load_path}")
        else:
            raise FileNotFoundError(f"Model files not found in {load_path}")
    
    # NEW: Probability target generation methods
    @handles_errors(default_return={}, context="generate_probability_targets")
    def generate_probability_targets(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        market_data: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Generate probability targets for multi-output training.
        
        Args:
            X: Feature matrix
            y: Target values
            market_data: Market data with OHLCV information
            
        Returns:
            Dictionary containing all 4 probability targets
        """
        if not self.config.enable_probability_outputs:
            self.logger.warning("Probability outputs not enabled in config")
            return {}
        
        self.logger.info("🔧 Generating probability targets for multi-output training")
        return self.probability_target_generator.generate_all_targets(X, y, market_data)
    
    @handles_errors(default_return={}, context="train_with_probability_targets")
    def train_with_probability_targets(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        market_data: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Train multi-output model with probability targets.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            market_data: Market data for probability target generation
            feature_names: List of feature names
            
        Returns:
            Dictionary containing trained models and metadata
        """
        if not self.config.enable_probability_outputs:
            self.logger.warning("Probability outputs not enabled, using standard training")
            return self._train_standard_multi_output(X_train, X_val, y_train, y_val, feature_names)
        
        self.logger.info("🔧 Training multi-output model with probability targets")
        
        # Generate probability targets
        y_train_prob = self.generate_probability_targets(X_train, y_train, market_data.iloc[:len(X_train)])
        y_val_prob = self.generate_probability_targets(X_val, y_val, market_data.iloc[len(X_train):])
        
        # Train models for each probability target
        trained_models = {}
        probability_metrics = {}
        
        for prob_type in self.config.probability_targets:
            self.logger.info(f"🔧 Training model for {prob_type}")
            
            # Get target values
            y_train_target = y_train_prob[prob_type.replace('_probability', '')]
            y_val_target = y_val_prob[prob_type.replace('_probability', '')]
            
            # Train model based on config using existing architectures
            if self.config.model_type == "LightGBM":
                model = self._train_lightgbm_probability_model(
                    X_train, X_val, y_train_target, y_val_target, feature_names, prob_type
                )
            elif self.config.model_type == "RandomForest":
                model = self._train_randomforest_probability_model(
                    X_train, X_val, y_train_target, y_val_target, feature_names, prob_type
                )
            elif self.config.model_type == "CNN" and EXISTING_MODELS_AVAILABLE:
                model = self._train_cnn_probability_model(
                    X_train, X_val, y_train_target, y_val_target, feature_names, prob_type
                )
            elif self.config.model_type == "TCN" and EXISTING_MODELS_AVAILABLE:
                model = self._train_tcn_probability_model(
                    X_train, X_val, y_train_target, y_val_target, feature_names, prob_type
                )
            elif self.config.model_type == "Transformer" and EXISTING_MODELS_AVAILABLE:
                model = self._train_transformer_probability_model(
                    X_train, X_val, y_train_target, y_val_target, feature_names, prob_type
                )
            else:
                self.logger.warning(f"Model type {self.config.model_type} not supported for probability training")
                continue
            
            trained_models[prob_type] = model
            probability_metrics[prob_type] = model.get("metrics", {})
        
        # Generate probability outputs
        probability_outputs = self._generate_probability_outputs(trained_models, X_val, market_data.iloc[len(X_train):])
        
        return {
            "trained_models": trained_models,
            "probability_metrics": probability_metrics,
            "probability_outputs": probability_outputs,
            "model_type": f"MultiOutput_{self.config.model_type}",
            "config": self.config.__dict__
        }
    
    def _train_lightgbm_probability_model(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        prob_type: str
    ) -> Dict[str, Any]:
        """Train LightGBM model for specific probability target using existing architecture."""
        self.logger.info(f"🔧 Training LightGBM for {prob_type}")
        
        # Use existing LightGBM configuration from step06 (Analyst model)
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=8,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        
        # Handle class imbalance
        try:
            from sklearn.utils.class_weight import compute_class_weight
            
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train), 
                y=y_train
            )
            sample_weights = class_weights[y_train.astype(int)]
            model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
        except Exception as e:
            self.logger.warning(f"Could not compute class weights for {prob_type}: {e}")
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
        
        # Evaluate
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred)
        }
        
        return {
            "model": model,
            "metrics": metrics,
            "feature_importance": model.feature_importances_,
            "prob_type": prob_type
        }
    
    def _train_randomforest_probability_model(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        prob_type: str
    ) -> Dict[str, Any]:
        """Train RandomForest model for specific probability target using existing architecture."""
        self.logger.info(f"🔧 Training RandomForest for {prob_type}")
        
        # Use existing RandomForest configuration from step09 (Tactician model)
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred)
        }
        
        return {
            "model": model,
            "metrics": metrics,
            "feature_importance": model.feature_importances_,
            "prob_type": prob_type
        }
    
    def _train_cnn_probability_model(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        prob_type: str
    ) -> Dict[str, Any]:
        """Train CNN model for specific probability target using existing architecture."""
        self.logger.info(f"🔧 Training CNN for {prob_type}")
        
        # Use existing CNN configuration from step06 (Tactician model)
        sequence_length = 32  # 32 periods (32 minutes of 1m data)
        X_train_sequences = self._create_sequences(X_train, sequence_length)
        X_val_sequences = self._create_sequences(X_val, sequence_length)
        
        # Adjust targets for sequence length
        y_train_seq = y_train[sequence_length:]
        y_val_seq = y_val[sequence_length:]
        
        # Create CNN model using existing architecture
        model = CNNModel(
            input_size=X_train.shape[1],
            sequence_length=sequence_length,
            num_classes=2,  # Binary classification
        )
        
        # Train model
        trainer = CNNTrainer(model, learning_rate=0.001, batch_size=32)
        history = trainer.train(X_train_sequences, y_train_seq, X_val_sequences, y_val_seq, epochs=100)
        
        # Evaluate
        y_pred = model.predict(X_val_sequences)
        y_pred_proba = model.predict_proba(X_val_sequences)
        
        metrics = {
            "accuracy": accuracy_score(y_val_seq, y_pred),
            "f1": f1_score(y_val_seq, y_pred),
            "precision": precision_score(y_val_seq, y_pred),
            "recall": recall_score(y_val_seq, y_pred)
        }
        
        return {
            "model": model,
            "metrics": metrics,
            "history": history,
            "prob_type": prob_type
        }
    
    def _train_tcn_probability_model(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        prob_type: str
    ) -> Dict[str, Any]:
        """Train TCN model for specific probability target using existing architecture."""
        self.logger.info(f"🔧 Training TCN for {prob_type}")
        
        # Use existing TCN configuration from step06 (Analyst model)
        sequence_length = 64  # 64 periods (16 hours of 15m data)
        X_train_sequences = self._create_sequences(X_train, sequence_length)
        X_val_sequences = self._create_sequences(X_val, sequence_length)
        
        # Adjust targets for sequence length
        y_train_seq = y_train[sequence_length:]
        y_val_seq = y_val[sequence_length:]
        
        # Create TCN model using existing architecture
        model = TCNModel(
            input_size=X_train.shape[1],
            sequence_length=sequence_length,
            num_classes=2,  # Binary classification
        )
        
        # Train model
        trainer = TCNTrainer(model, learning_rate=0.0001, batch_size=32)
        history = trainer.train(X_train_sequences, y_train_seq, X_val_sequences, y_val_seq, epochs=150)
        
        # Evaluate
        y_pred = model.predict(X_val_sequences)
        y_pred_proba = model.predict_proba(X_val_sequences)
        
        metrics = {
            "accuracy": accuracy_score(y_val_seq, y_pred),
            "f1": f1_score(y_val_seq, y_pred),
            "precision": precision_score(y_val_seq, y_pred),
            "recall": recall_score(y_val_seq, y_pred)
        }
        
        return {
            "model": model,
            "metrics": metrics,
            "history": history,
            "prob_type": prob_type
        }
    
    def _train_transformer_probability_model(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        prob_type: str
    ) -> Dict[str, Any]:
        """Train Transformer model for specific probability target using existing architecture."""
        self.logger.info(f"🔧 Training Transformer for {prob_type}")
        
        # Use existing Transformer configuration from step06 (Analyst model)
        sequence_length = 16  # 16 periods (4 hours of 15m data)
        X_train_sequences = self._create_sequences(X_train, sequence_length)
        X_val_sequences = self._create_sequences(X_val, sequence_length)
        
        # Adjust targets for sequence length
        y_train_seq = y_train[sequence_length:]
        y_val_seq = y_val[sequence_length:]
        
        # Create Transformer model using existing architecture
        model = TransformerModel(
            input_size=X_train.shape[1],
            d_model=256,
            nhead=8,
            num_layers=6,
            num_classes=2,  # Binary classification
        )
        
        # Train model
        trainer = TransformerTrainer(model, learning_rate=0.0001, batch_size=32)
        history = trainer.train(X_train_sequences, y_train_seq, X_val_sequences, y_val_seq, epochs=150)
        
        # Evaluate
        y_pred = model.predict(X_val_sequences)
        y_pred_proba = model.predict_proba(X_val_sequences)
        
        metrics = {
            "accuracy": accuracy_score(y_val_seq, y_pred),
            "f1": f1_score(y_val_seq, y_pred),
            "precision": precision_score(y_val_seq, y_pred),
            "recall": recall_score(y_val_seq, y_pred)
        }
        
        return {
            "model": model,
            "metrics": metrics,
            "history": history,
            "prob_type": prob_type
        }
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences for time series models."""
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    def _generate_probability_outputs(
        self,
        trained_models: Dict[str, Any],
        X_test: np.ndarray,
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Generate probability outputs from trained models."""
        probabilities = {}
        
        for prob_type, model_info in trained_models.items():
            model = model_info["model"]
            
            try:
                # Get probability predictions
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)
                    if proba.shape[1] > 1:
                        # Binary classification, get positive class probability
                        prob_value = proba[:, 1].mean()
                    else:
                        # Single class, use the probability
                        prob_value = proba[:, 0].mean()
                else:
                    # Fallback to prediction
                    pred = model.predict(X_test)
                    prob_value = pred.mean()
                
                # Ensure probability is in [0, 1] range
                prob_value = np.clip(prob_value, 0.0, 1.0)
                
                probabilities[prob_type] = float(prob_value)
                
            except Exception as e:
                self.logger.error(f"Error predicting {prob_type} probability: {e}")
                probabilities[prob_type] = 0.5
        
        # Add metadata
        probabilities["generation_timestamp"] = datetime.now().isoformat()
        probabilities["model_type"] = "multi_output"
        
        return probabilities
    
    def _train_standard_multi_output(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Fallback to standard multi-output training."""
        self.logger.info("🔧 Using standard multi-output training")
        
        # This would call the existing training methods
        # For now, return a placeholder
        return {
            "model_type": "standard_multi_output",
            "note": "Standard training used (probability outputs disabled)"
        }

def create_multi_output_trainer(
    model_type: str = "LightGBM",
    use_profit_features: bool = True,
    **kwargs
) -> MultiOutputModelTrainer:
    """Factory function to create a multi-output model trainer.
    
    Args:
        model_type: Type of model to use ("LightGBM", "RandomForest", "NeuralNetwork")
        use_profit_features: Whether to use profit-based feature engineering
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured MultiOutputModelTrainer instance
    """
    config = MultiOutputModelConfig(
        model_type=model_type,
        use_profit_features=use_profit_features,
        **kwargs
    )
    
    return MultiOutputModelTrainer(config)