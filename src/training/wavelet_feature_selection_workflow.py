"""
Wavelet Feature Selection Workflow

This module implements a comprehensive workflow using the two-model strategy:
1. Discovery Model: Trained on full feature set to identify winning features
2. Production Model: Trained on lean feature set for live deployment

The workflow:
1. Run full, extensive wavelet analysis (as in backtesting/training)
2. Build Discovery Model using the rich feature set
3. Perform feature selection using permutation importance and SHAP
4. Identify the most important features
5. Create lean dataset with only winning features
6. Train Production Model on lean dataset
7. Create optimized live trading configurations
"""

import asyncio
import pandas as pd
import numpy as np
import yaml
import time
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import shap

from src.training.steps.vectorized_advanced_feature_engineering import VectorizedAdvancedFeatureEngineering
from src.training.steps.precompute_wavelet_features import WaveletFeaturePrecomputer
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


@dataclass
class FeatureImportanceResult:
    """Container for feature importance analysis results."""
    feature_name: str
    permutation_importance: float
    shap_importance: float
    combined_score: float
    feature_type: str  # 'wavelet', 'technical', 'other'
    computation_cost: float  # Estimated computation time in ms


class WaveletFeatureSelectionWorkflow:
    """
    Comprehensive workflow for wavelet feature selection using two-model strategy.
    
    This workflow:
    1. Runs full wavelet analysis with all features
    2. Builds Discovery Model on the rich feature set
    3. Performs feature selection using multiple methods
    4. Identifies the most important features
    5. Creates lean dataset with only winning features
    6. Trains Production Model on lean dataset
    7. Creates optimized live trading configurations
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("WaveletFeatureSelectionWorkflow")
        
        # Workflow configuration
        self.workflow_config = config.get("wavelet_feature_selection", {})
        self.output_dir = Path(self.workflow_config.get("output_dir", "data/wavelet_feature_selection"))
        self.model_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"
        self.configs_dir = self.output_dir / "configs"
        
        # Feature selection parameters
        self.top_n_features = self.workflow_config.get("top_n_features", 20)
        self.min_importance_threshold = self.workflow_config.get("min_importance_threshold", 0.01)
        self.max_computation_time = self.workflow_config.get("max_computation_time", 0.1)  # 100ms
        
        # ML model parameters
        self.test_size = self.workflow_config.get("test_size", 0.2)
        self.random_state = self.workflow_config.get("random_state", 42)
        self.cv_folds = self.workflow_config.get("cv_folds", 5)
        
        # Model configurations
        self.discovery_model_config = self.workflow_config.get("discovery_model", {
            "type": "random_forest",
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2
        })
        
        self.production_model_config = self.workflow_config.get("production_model", {
            "type": "gradient_boosting",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8
        })
        
        # Initialize components
        self.feature_engineer: Optional[VectorizedAdvancedFeatureEngineering] = None
        self.feature_precomputer: Optional[WaveletFeaturePrecomputer] = None
        
        # Results storage
        self.feature_importance_results: List[FeatureImportanceResult] = []
        self.discovery_model_performance: Dict[str, Any] = {}
        self.production_model_performance: Dict[str, Any] = {}
        self.optimized_configs: Dict[str, Any] = {}
        
        # Models
        self.discovery_model: Optional[Any] = None
        self.production_model: Optional[Any] = None
        
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="wavelet feature selection workflow initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the wavelet feature selection workflow."""
        try:
            self.logger.info("🚀 Initializing Wavelet Feature Selection Workflow...")
            
            # Create output directories
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir.mkdir(exist_ok=True)
            self.results_dir.mkdir(exist_ok=True)
            self.configs_dir.mkdir(exist_ok=True)
            
            # Initialize feature engineering with full configuration
            self.feature_engineer = VectorizedAdvancedFeatureEngineering(self.config)
            success = await self.feature_engineer.initialize()
            if not success:
                self.logger.error("Failed to initialize feature engineer")
                return False
            
            # Initialize feature precomputer
            self.feature_precomputer = WaveletFeaturePrecomputer(self.config)
            success = await self.feature_precomputer.initialize()
            if not success:
                self.logger.error("Failed to initialize feature precomputer")
                return False
            
            self.logger.info("✅ Wavelet Feature Selection Workflow initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing workflow: {e}")
            return False
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="full wavelet analysis execution",
    )
    async def run_full_wavelet_analysis(
        self, 
        price_data: pd.DataFrame, 
        volume_data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Step 1: Run full, extensive wavelet analysis as in backtesting/training.
        
        Args:
            price_data: OHLCV price data
            volume_data: Volume data
            
        Returns:
            Dictionary containing all wavelet features
        """
        try:
            self.logger.info("📊 Step 1: Running full wavelet analysis...")
            start_time = time.time()
            
            # Run full feature engineering with all wavelet features
            features = await self.feature_engineer.engineer_features(
                price_data, volume_data
            )
            
            # Extract wavelet features
            wavelet_features = {k: v for k, v in features.items() if 'wavelet' in k.lower()}
            
            computation_time = time.time() - start_time
            self.logger.info(f"✅ Full wavelet analysis completed in {computation_time:.2f}s")
            self.logger.info(f"📊 Generated {len(wavelet_features)} wavelet features")
            
            return {
                "all_features": features,
                "wavelet_features": wavelet_features,
                "computation_time": computation_time,
                "feature_count": len(features),
                "wavelet_feature_count": len(wavelet_features)
            }
            
        except Exception as e:
            self.logger.error(f"Error in full wavelet analysis: {e}")
            return None
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="discovery model training",
    )
    async def train_discovery_model(
        self, 
        features: Dict[str, Any], 
        labels: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """
        Step 2: Train Discovery Model using the rich feature set.
        
        Args:
            features: All engineered features
            labels: Target labels for prediction
            
        Returns:
            Dictionary containing trained discovery model and performance metrics
        """
        try:
            self.logger.info("🔍 Step 2: Training Discovery Model...")
            
            # Prepare feature matrix
            feature_df = pd.DataFrame(features)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                feature_df, labels, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=labels
            )
            
            # Train Discovery Model (optimized for feature selection)
            model_type = self.discovery_model_config.get("type", "random_forest")
            
            if model_type == "random_forest":
                discovery_model = RandomForestClassifier(
                    n_estimators=self.discovery_model_config.get("n_estimators", 200),
                    max_depth=self.discovery_model_config.get("max_depth", 15),
                    min_samples_split=self.discovery_model_config.get("min_samples_split", 5),
                    min_samples_leaf=self.discovery_model_config.get("min_samples_leaf", 2),
                    random_state=self.random_state,
                    n_jobs=-1
                )
            elif model_type == "gradient_boosting":
                discovery_model = GradientBoostingClassifier(
                    n_estimators=self.discovery_model_config.get("n_estimators", 200),
                    max_depth=self.discovery_model_config.get("max_depth", 15),
                    learning_rate=self.discovery_model_config.get("learning_rate", 0.1),
                    random_state=self.random_state
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train the model
            discovery_model.fit(X_train, y_train)
            self.discovery_model = discovery_model
            
            # Evaluate Discovery Model
            cv_scores = cross_val_score(discovery_model, X_train, y_train, cv=self.cv_folds)
            y_pred = discovery_model.predict(X_test)
            
            performance = {
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "test_accuracy": (y_pred == y_test).mean(),
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            }
            
            self.discovery_model_performance = performance
            
            # Save Discovery Model (for analysis purposes)
            model_path = self.model_dir / "discovery_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(discovery_model, f)
            
            self.logger.info("✅ Discovery Model trained successfully")
            self.logger.info(f"📊 Discovery Model Performance:")
            self.logger.info(f"  CV Score: {performance['cv_mean']:.3f} ± {performance['cv_std']:.3f}")
            self.logger.info(f"  Test Accuracy: {performance['test_accuracy']:.3f}")
            
            return {
                "model": discovery_model,
                "performance": performance,
                "feature_names": list(feature_df.columns),
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test
            }
            
        except Exception as e:
            self.logger.error(f"Error training discovery model: {e}")
            return None
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="feature importance analysis",
    )
    async def perform_feature_selection(
        self, 
        discovery_model_data: Dict[str, Any]
    ) -> Optional[List[FeatureImportanceResult]]:
        """
        Step 3: Perform feature selection using permutation importance and SHAP.
        
        Args:
            discovery_model_data: Results from discovery model training
            
        Returns:
            List of feature importance results
        """
        try:
            self.logger.info("🔍 Step 3: Performing feature selection...")
            
            model = discovery_model_data["model"]
            X_test = discovery_model_data["X_test"]
            y_test = discovery_model_data["y_test"]
            feature_names = discovery_model_data["feature_names"]
            
            results = []
            
            # Permutation Importance
            self.logger.info("📊 Computing permutation importance...")
            perm_importance = permutation_importance(
                model, X_test, y_test,
                n_repeats=10, random_state=self.random_state
            )
            
            # SHAP Analysis
            self.logger.info("📊 Computing SHAP importance...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Calculate SHAP importance (mean absolute SHAP values)
            if len(shap_values.shape) == 3:  # Multi-class
                shap_importance = np.mean(np.abs(shap_values), axis=(0, 1))
            else:  # Binary
                shap_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Combine results
            for i, feature_name in enumerate(feature_names):
                # Determine feature type
                feature_type = self._classify_feature_type(feature_name)
                
                # Estimate computation cost
                computation_cost = self._estimate_computation_cost(feature_name)
                
                # Calculate combined score
                perm_score = perm_importance.importances_mean[i]
                shap_score = shap_importance[i]
                combined_score = (perm_score + shap_score) / 2
                
                result = FeatureImportanceResult(
                    feature_name=feature_name,
                    permutation_importance=perm_score,
                    shap_importance=shap_score,
                    combined_score=combined_score,
                    feature_type=feature_type,
                    computation_cost=computation_cost
                )
                results.append(result)
            
            # Sort by combined score
            results.sort(key=lambda x: x.combined_score, reverse=True)
            
            self.feature_importance_results = results
            
            self.logger.info(f"✅ Feature selection completed. Top features:")
            for i, result in enumerate(results[:10]):
                self.logger.info(f"  {i+1}. {result.feature_name}: {result.combined_score:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return None
    
    def _classify_feature_type(self, feature_name: str) -> str:
        """Classify feature type based on name."""
        feature_name_lower = feature_name.lower()
        
        if 'wavelet' in feature_name_lower:
            return 'wavelet'
        elif any(x in feature_name_lower for x in ['rsi', 'macd', 'sma', 'ema', 'bollinger']):
            return 'technical'
        else:
            return 'other'
    
    def _estimate_computation_cost(self, feature_name: str) -> float:
        """Estimate computation cost in milliseconds."""
        feature_name_lower = feature_name.lower()
        
        # Base costs for different feature types
        if 'wavelet' in feature_name_lower:
            if 'cwt' in feature_name_lower:
                return 50.0  # Continuous wavelet is expensive
            elif 'packet' in feature_name_lower:
                return 40.0  # Wavelet packets are expensive
            elif 'denoising' in feature_name_lower:
                return 30.0  # Denoising is moderate
            else:
                return 10.0  # Basic DWT is fast
        elif 'technical' in feature_name_lower:
            return 1.0  # Technical indicators are fast
        else:
            return 5.0  # Other features are moderate
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="winner feature identification",
    )
    async def identify_winner_features(self) -> Optional[List[FeatureImportanceResult]]:
        """
        Step 4: Identify the most important features for live trading.
        
        Returns:
            List of winner features optimized for live trading
        """
        try:
            self.logger.info("🏆 Step 4: Identifying winner features...")
            
            if not self.feature_importance_results:
                self.logger.error("No feature importance results available")
                return None
            
            # Filter features based on importance and computation cost
            winners = []
            total_computation_time = 0.0
            
            for result in self.feature_importance_results:
                # Check importance threshold
                if result.combined_score < self.min_importance_threshold:
                    continue
                
                # Check computation time constraint
                if total_computation_time + result.computation_cost > self.max_computation_time * 1000:  # Convert to ms
                    continue
                
                # Add to winners
                winners.append(result)
                total_computation_time += result.computation_cost
                
                # Limit to top N features
                if len(winners) >= self.top_n_features:
                    break
            
            self.logger.info(f"✅ Identified {len(winners)} winner features")
            self.logger.info(f"📊 Total computation time: {total_computation_time:.1f}ms")
            
            # Log winner features
            for i, winner in enumerate(winners):
                self.logger.info(f"  {i+1}. {winner.feature_name}: "
                               f"score={winner.combined_score:.4f}, "
                               f"cost={winner.computation_cost:.1f}ms")
            
            return winners
            
        except Exception as e:
            self.logger.error(f"Error identifying winner features: {e}")
            return None
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="lean dataset creation",
    )
    async def create_lean_dataset(
        self, 
        winner_features: List[FeatureImportanceResult],
        original_features: Dict[str, Any],
        labels: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """
        Step 5: Create lean dataset with only winning features.
        
        Args:
            winner_features: List of identified winner features
            original_features: Original full feature set
            labels: Target labels
            
        Returns:
            Dictionary containing lean dataset
        """
        try:
            self.logger.info("📊 Step 5: Creating lean dataset...")
            
            # Extract only winning features
            winner_feature_names = [f.feature_name for f in winner_features]
            lean_features = {name: original_features[name] for name in winner_feature_names}
            
            # Create lean feature matrix
            lean_feature_df = pd.DataFrame(lean_features)
            
            # Split lean dataset
            X_train_lean, X_test_lean, y_train_lean, y_test_lean = train_test_split(
                lean_feature_df, labels, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=labels
            )
            
            self.logger.info(f"✅ Lean dataset created with {len(winner_features)} features")
            self.logger.info(f"📊 Lean dataset shape: {lean_feature_df.shape}")
            
            return {
                "lean_features": lean_features,
                "lean_feature_df": lean_feature_df,
                "X_train_lean": X_train_lean,
                "X_test_lean": X_test_lean,
                "y_train_lean": y_train_lean,
                "y_test_lean": y_test_lean,
                "winner_feature_names": winner_feature_names
            }
            
        except Exception as e:
            self.logger.error(f"Error creating lean dataset: {e}")
            return None
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="production model training",
    )
    async def train_production_model(
        self, 
        lean_dataset: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Step 6: Train Production Model on lean dataset.
        
        Args:
            lean_dataset: Lean dataset with only winning features
            
        Returns:
            Dictionary containing trained production model and performance metrics
        """
        try:
            self.logger.info("🚀 Step 6: Training Production Model...")
            
            X_train_lean = lean_dataset["X_train_lean"]
            X_test_lean = lean_dataset["X_test_lean"]
            y_train_lean = lean_dataset["y_train_lean"]
            y_test_lean = lean_dataset["y_test_lean"]
            
            # Train Production Model (optimized for deployment)
            model_type = self.production_model_config.get("type", "gradient_boosting")
            
            if model_type == "gradient_boosting":
                production_model = GradientBoostingClassifier(
                    n_estimators=self.production_model_config.get("n_estimators", 100),
                    max_depth=self.production_model_config.get("max_depth", 6),
                    learning_rate=self.production_model_config.get("learning_rate", 0.1),
                    subsample=self.production_model_config.get("subsample", 0.8),
                    random_state=self.random_state
                )
            elif model_type == "random_forest":
                production_model = RandomForestClassifier(
                    n_estimators=self.production_model_config.get("n_estimators", 100),
                    max_depth=self.production_model_config.get("max_depth", 6),
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train the production model
            production_model.fit(X_train_lean, y_train_lean)
            self.production_model = production_model
            
            # Evaluate Production Model
            cv_scores = cross_val_score(production_model, X_train_lean, y_train_lean, cv=self.cv_folds)
            y_pred_lean = production_model.predict(X_test_lean)
            
            performance = {
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "test_accuracy": (y_pred_lean == y_test_lean).mean(),
                "classification_report": classification_report(y_test_lean, y_pred_lean, output_dict=True)
            }
            
            self.production_model_performance = performance
            
            # Save Production Model (for deployment)
            model_path = self.model_dir / "production_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(production_model, f)
            
            # Save lean feature names for deployment
            feature_names_path = self.model_dir / "production_features.json"
            import json
            with open(feature_names_path, 'w') as f:
                json.dump(lean_dataset["winner_feature_names"], f)
            
            self.logger.info("✅ Production Model trained successfully")
            self.logger.info(f"📊 Production Model Performance:")
            self.logger.info(f"  CV Score: {performance['cv_mean']:.3f} ± {performance['cv_std']:.3f}")
            self.logger.info(f"  Test Accuracy: {performance['test_accuracy']:.3f}")
            
            return {
                "model": production_model,
                "performance": performance,
                "feature_names": lean_dataset["winner_feature_names"],
                "X_train_lean": X_train_lean,
                "X_test_lean": X_test_lean,
                "y_train_lean": y_train_lean,
                "y_test_lean": y_test_lean
            }
            
        except Exception as e:
            self.logger.error(f"Error training production model: {e}")
            return None
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="live configuration creation",
    )
    async def create_live_configurations(
        self, 
        winner_features: List[FeatureImportanceResult],
        production_model_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Step 7: Create optimized live trading configurations.
        
        Args:
            winner_features: List of identified winner features
            production_model_data: Production model results
            
        Returns:
            Dictionary containing optimized configurations
        """
        try:
            self.logger.info("⚡ Step 7: Creating live configurations...")
            
            # Group features by type
            wavelet_features = [f for f in winner_features if f.feature_type == 'wavelet']
            technical_features = [f for f in winner_features if f.feature_type == 'technical']
            other_features = [f for f in winner_features if f.feature_type == 'other']
            
            # Create optimized wavelet configuration
            optimized_wavelet_config = self._create_optimized_wavelet_config(wavelet_features)
            
            # Create live trading configuration
            live_config = self._create_live_trading_config(winner_features, production_model_data)
            
            # Create production model configuration
            production_config = self._create_production_model_config(production_model_data)
            
            # Create performance configuration
            performance_config = self._create_performance_config(winner_features)
            
            # Save configurations
            configs = {
                "optimized_wavelet": optimized_wavelet_config,
                "live_trading": live_config,
                "production_model": production_config,
                "performance": performance_config
            }
            
            for name, config in configs.items():
                config_path = self.configs_dir / f"{name}_config.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            
            self.optimized_configs = configs
            
            self.logger.info("✅ Live configurations created successfully")
            self.logger.info(f"📊 Wavelet features: {len(wavelet_features)}")
            self.logger.info(f"📊 Technical features: {len(technical_features)}")
            self.logger.info(f"📊 Other features: {len(other_features)}")
            
            return configs
            
        except Exception as e:
            self.logger.error(f"Error creating live configurations: {e}")
            return None
    
    def _create_optimized_wavelet_config(self, wavelet_features: List[FeatureImportanceResult]) -> Dict[str, Any]:
        """Create optimized wavelet configuration based on winner features."""
        config = {
            "wavelet_transforms": {
                "wavelet_type": "db4",  # Single type for speed
                "decomposition_level": 2,  # Minimal levels
                "padding_mode": "symmetric",
                "enable_discrete_wavelet": True,
                "enable_continuous_wavelet": False,  # Disable expensive CWT
                "enable_wavelet_packet": False,  # Disable expensive packets
                "enable_denoising": False,  # Disable expensive denoising
                "max_wavelet_types": 1,  # Single type only
                "enable_stationary_series": True,
                "stationary_transforms": ["price_diff"],  # Only price differences
                "max_features_per_wavelet": len(wavelet_features),
                "feature_selection_method": "importance",
                "selected_features": [f.feature_name for f in wavelet_features]
            },
            "performance_constraints": {
                "max_computation_time": 0.1,  # 100ms
                "max_data_points": 256,
                "sliding_window_size": 128
            }
        }
        
        return config
    
    def _create_live_trading_config(self, winner_features: List[FeatureImportanceResult], production_model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create live trading configuration."""
        config = {
            "live_wavelet_analyzer": {
                "max_computation_time": 0.1,
                "sliding_window_size": 128,
                "wavelet_type": "db4",
                "decomposition_level": 2,
                "energy_threshold": 0.01,
                "entropy_threshold": 0.5,
                "confidence_threshold": 0.7
            },
            "feature_selection": {
                "selected_features": [f.feature_name for f in winner_features],
                "feature_weights": {f.feature_name: f.combined_score for f in winner_features},
                "total_computation_cost": sum(f.computation_cost for f in winner_features)
            },
            "production_model": {
                "model_path": "data/wavelet_feature_selection/models/production_model.pkl",
                "feature_names_path": "data/wavelet_feature_selection/models/production_features.json",
                "model_type": self.production_model_config.get("type", "gradient_boosting"),
                "performance": production_model_data["performance"]
            },
            "signal_generation": {
                "enable_wavelet_signals": True,
                "wavelet_signal_weight": 0.3,
                "min_confidence": 0.6,
                "max_signal_age": 60
            }
        }
        
        return config
    
    def _create_production_model_config(self, production_model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create production model configuration."""
        config = {
            "model_info": {
                "model_type": self.production_model_config.get("type", "gradient_boosting"),
                "feature_count": len(production_model_data["feature_names"]),
                "model_path": "data/wavelet_feature_selection/models/production_model.pkl",
                "feature_names_path": "data/wavelet_feature_selection/models/production_features.json"
            },
            "performance": {
                "cv_mean": production_model_data["performance"]["cv_mean"],
                "cv_std": production_model_data["performance"]["cv_std"],
                "test_accuracy": production_model_data["performance"]["test_accuracy"]
            },
            "deployment": {
                "enable_model_loading": True,
                "model_cache_size": 1,
                "prediction_timeout": 0.05,  # 50ms
                "enable_feature_validation": True
            }
        }
        
        return config
    
    def _create_performance_config(self, winner_features: List[FeatureImportanceResult]) -> Dict[str, Any]:
        """Create performance monitoring configuration."""
        config = {
            "performance_monitoring": {
                "enable_performance_tracking": True,
                "target_computation_time": 0.05,  # 50ms target
                "max_computation_time": 0.1,  # 100ms limit
                "target_signal_accuracy": 0.6,
                "min_signal_rate": 0.01
            },
            "feature_analysis": {
                "total_features": len(winner_features),
                "wavelet_features": len([f for f in winner_features if f.feature_type == 'wavelet']),
                "technical_features": len([f for f in winner_features if f.feature_type == 'technical']),
                "other_features": len([f for f in winner_features if f.feature_type == 'other']),
                "total_computation_cost": sum(f.computation_cost for f in winner_features)
            }
        }
        
        return config
    
    async def run_complete_workflow(
        self, 
        price_data: pd.DataFrame, 
        volume_data: pd.DataFrame, 
        labels: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """
        Run the complete wavelet feature selection workflow using two-model strategy.
        
        Args:
            price_data: OHLCV price data
            volume_data: Volume data
            labels: Target labels for prediction
            
        Returns:
            Complete workflow results
        """
        try:
            self.logger.info("🚀 Starting complete wavelet feature selection workflow...")
            
            # Step 1: Full wavelet analysis
            analysis_results = await self.run_full_wavelet_analysis(price_data, volume_data)
            if not analysis_results:
                return None
            
            # Step 2: Train Discovery Model
            discovery_model_results = await self.train_discovery_model(analysis_results["all_features"], labels)
            if not discovery_model_results:
                return None
            
            # Step 3: Feature selection
            feature_results = await self.perform_feature_selection(discovery_model_results)
            if not feature_results:
                return None
            
            # Step 4: Identify winners
            winner_features = await self.identify_winner_features()
            if not winner_features:
                return None
            
            # Step 5: Create lean dataset
            lean_dataset = await self.create_lean_dataset(winner_features, analysis_results["all_features"], labels)
            if not lean_dataset:
                return None
            
            # Step 6: Train Production Model
            production_model_results = await self.train_production_model(lean_dataset)
            if not production_model_results:
                return None
            
            # Step 7: Create live configurations
            live_configs = await self.create_live_configurations(winner_features, production_model_results)
            if not live_configs:
                return None
            
            # Generate summary report
            summary = self._generate_summary_report(
                analysis_results, discovery_model_results, feature_results, 
                winner_features, production_model_results, live_configs
            )
            
            self.logger.info("✅ Complete workflow finished successfully!")
            
            return {
                "analysis_results": analysis_results,
                "discovery_model_results": discovery_model_results,
                "feature_results": feature_results,
                "winner_features": winner_features,
                "lean_dataset": lean_dataset,
                "production_model_results": production_model_results,
                "live_configs": live_configs,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error in complete workflow: {e}")
            return None
    
    def _generate_summary_report(
        self, 
        analysis_results: Dict[str, Any],
        discovery_model_results: Dict[str, Any],
        feature_results: List[FeatureImportanceResult],
        winner_features: List[FeatureImportanceResult],
        production_model_results: Dict[str, Any],
        live_configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        summary = {
            "workflow_summary": {
                "total_features_generated": analysis_results["feature_count"],
                "wavelet_features_generated": analysis_results["wavelet_feature_count"],
                "computation_time_full_analysis": analysis_results["computation_time"],
                "discovery_model_accuracy": discovery_model_results["performance"]["test_accuracy"],
                "production_model_accuracy": production_model_results["performance"]["test_accuracy"],
                "features_analyzed": len(feature_results),
                "winner_features_selected": len(winner_features),
                "total_computation_cost_ms": sum(f.computation_cost for f in winner_features)
            },
            "model_comparison": {
                "discovery_model": {
                    "cv_mean": discovery_model_results["performance"]["cv_mean"],
                    "cv_std": discovery_model_results["performance"]["cv_std"],
                    "test_accuracy": discovery_model_results["performance"]["test_accuracy"]
                },
                "production_model": {
                    "cv_mean": production_model_results["performance"]["cv_mean"],
                    "cv_std": production_model_results["performance"]["cv_std"],
                    "test_accuracy": production_model_results["performance"]["test_accuracy"]
                }
            },
            "performance_improvement": {
                "computation_time_reduction": (
                    analysis_results["computation_time"] - 
                    (sum(f.computation_cost for f in winner_features) / 1000)
                ) / analysis_results["computation_time"],
                "feature_count_reduction": (
                    analysis_results["feature_count"] - len(winner_features)
                ) / analysis_results["feature_count"],
                "accuracy_preservation": (
                    production_model_results["performance"]["test_accuracy"] /
                    discovery_model_results["performance"]["test_accuracy"]
                )
            },
            "winner_features": [
                {
                    "name": f.feature_name,
                    "importance_score": f.combined_score,
                    "computation_cost_ms": f.computation_cost,
                    "feature_type": f.feature_type
                }
                for f in winner_features
            ]
        }
        
        return summary