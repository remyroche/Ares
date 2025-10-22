"""
Enhanced Analyst Base Training - Comprehensive Tools Integration

This module provides an enhanced version of the Analyst base training component
that uses the generalized comprehensive tools from BaseStep.

Key Features:
- Uses GeneralizedModelTrainingBase for comprehensive tools access
- Enhanced data processing with comprehensive tools
- Advanced model management and persistence
- Performance monitoring and logging
- Error handling and recovery mechanisms
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from ..core.generalized_model_training_base import (
    GeneralizedModelTrainingBase, ModelTrainingConfig, ModelTrainingResult, 
    ModelTrainingRole, ModelType
)
from ..utils.comprehensive_tools_integration import (
    ComprehensiveToolsIntegration, ComprehensiveToolsConfig,
    with_comprehensive_tools, with_memory_optimization, with_performance_tracking
)
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_performance, tprint_data_format, LogLevel,
    tprint_operation_start, tprint_operation_end, tprint_data_preview,
    tprint_dict, tprint_list, tprint_dataframe_info, tprint_model_info,
    tprint_performance_summary, tprint_memory_usage, tprint_hardware_stats
)


@dataclass
class EnhancedAnalystTrainingConfig:
    """Enhanced configuration for Analyst base training."""
    # Model types
    model_types: List[str] = field(default_factory=lambda: ['lightgbm', 'catboost'])
    timeframe: str = "15m"
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    
    # Training parameters
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    random_seed: Optional[int] = None
    
    # Model-specific parameters
    lightgbm_params: Dict[str, Any] = field(default_factory=dict)
    catboost_params: Dict[str, Any] = field(default_factory=dict)
    
    # Feature engineering
    enable_patchtst_features: bool = True
    enable_regime_features: bool = True
    enable_multi_timeframe: bool = True
    
    # Comprehensive tools configuration
    enable_comprehensive_tools: bool = True
    enable_hardware_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_performance_monitoring: bool = True
    enable_detailed_logging: bool = True
    
    # Auto-save configuration
    auto_save: bool = True
    save_artifacts: bool = True


@dataclass
class EnhancedAnalystTrainingResult:
    """Enhanced result of Analyst base training."""
    success: bool
    models: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, Dict[str, float]]] = None
    artifacts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    comprehensive_tools_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedAnalystBaseTraining(GeneralizedModelTrainingBase):
    """
    Enhanced Analyst base training component using comprehensive tools.
    
    This component leverages all BaseStep comprehensive tools for enhanced
    data processing, model training, and performance monitoring.
    """
    
    def __init__(
        self,
        step_name: str = "enhanced_analyst_base_training",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the enhanced analyst base training component.
        
        Args:
            step_name: Name of the training step
            config: Configuration dictionary
            logger: Logger instance (optional)
        """
        # Parse enhanced configuration
        enhanced_config = self._parse_enhanced_config(config)
        
        # Convert to generalized model training config
        generalized_config = self._convert_to_generalized_config(enhanced_config)
        
        # Initialize with generalized base
        super().__init__(step_name, generalized_config, logger)
        
        # Store enhanced configuration
        self.enhanced_config = enhanced_config
        
        # Initialize comprehensive tools integration
        if enhanced_config.enable_comprehensive_tools:
            self.comprehensive_tools = ComprehensiveToolsIntegration(
                self, 
                ComprehensiveToolsConfig(
                    enable_logging=enhanced_config.enable_detailed_logging,
                    enable_performance_monitoring=enhanced_config.enable_performance_monitoring,
                    enable_memory_optimization=enhanced_config.enable_memory_optimization,
                    enable_hardware_optimization=enhanced_config.enable_hardware_optimization,
                    enable_error_handling=True,
                    log_level="INFO"
                )
            )
        
        tprint_banner("Enhanced Analyst Base Training")
        tprint_info("🔧 Initialized with comprehensive tools integration")
    
    def _parse_enhanced_config(self, config: Optional[Dict[str, Any]]) -> EnhancedAnalystTrainingConfig:
        """Parse enhanced configuration."""
        if not config:
            config = {}
        
        return EnhancedAnalystTrainingConfig(
            model_types=config.get('model_types', ['lightgbm', 'catboost']),
            timeframe=config.get('timeframe', '15m'),
            symbol=config.get('symbol', 'ETHUSDT'),
            exchange=config.get('exchange', 'binance'),
            validation_split=config.get('validation_split', 0.2),
            cross_validation_folds=config.get('cross_validation_folds', 5),
            random_seed=config.get('random_seed'),
            lightgbm_params=config.get('lightgbm_params', {}),
            catboost_params=config.get('catboost_params', {}),
            enable_patchtst_features=config.get('enable_patchtst_features', True),
            enable_regime_features=config.get('enable_regime_features', True),
            enable_multi_timeframe=config.get('enable_multi_timeframe', True),
            enable_comprehensive_tools=config.get('enable_comprehensive_tools', True),
            enable_hardware_optimization=config.get('enable_hardware_optimization', True),
            enable_memory_optimization=config.get('enable_memory_optimization', True),
            enable_performance_monitoring=config.get('enable_performance_monitoring', True),
            enable_detailed_logging=config.get('enable_detailed_logging', True),
            auto_save=config.get('auto_save', True),
            save_artifacts=config.get('save_artifacts', True)
        )
    
    def _convert_to_generalized_config(self, enhanced_config: EnhancedAnalystTrainingConfig) -> Dict[str, Any]:
        """Convert enhanced config to generalized config."""
        return {
            'role': 'analyst',
            'model_types': enhanced_config.model_types,
            'timeframe': enhanced_config.timeframe,
            'symbol': enhanced_config.symbol,
            'exchange': enhanced_config.exchange,
            'validation_split': enhanced_config.validation_split,
            'cross_validation_folds': enhanced_config.cross_validation_folds,
            'random_seed': enhanced_config.random_seed,
            'enable_hyperparameter_optimization': True,
            'enable_ensemble': True,
            'enable_early_stopping': True,
            'early_stopping_patience': 10,
            'enable_hardware_optimization': enhanced_config.enable_hardware_optimization,
            'enable_memory_optimization': enhanced_config.enable_memory_optimization,
            'enable_detailed_logging': enhanced_config.enable_detailed_logging,
            'enable_performance_monitoring': enhanced_config.enable_performance_monitoring,
            'enable_artifact_management': enhanced_config.save_artifacts,
            'custom_params': {
                'lightgbm_params': enhanced_config.lightgbm_params,
                'catboost_params': enhanced_config.catboost_params,
                'enable_patchtst_features': enhanced_config.enable_patchtst_features,
                'enable_regime_features': enhanced_config.enable_regime_features,
                'enable_multi_timeframe': enhanced_config.enable_multi_timeframe
            }
        }
    
    # ============================================================================
    # ENHANCED TRAINING METHODS
    # ============================================================================
    
    @with_comprehensive_tools(
        enable_logging=True,
        enable_performance_monitoring=True,
        enable_memory_optimization=True,
        enable_hardware_optimization=True,
        enable_error_handling=True
    )
    async def train_models(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> ModelTrainingResult:
        """
        Train Analyst models using comprehensive tools.
        
        Args:
            data: Training data
            targets: Target variables (optional)
            
        Returns:
            Model training result
        """
        try:
            tprint_operation_start("Enhanced Analyst Model Training")
            
            # 1. Data preprocessing with comprehensive tools
            tprint_info("📊 Step 1: Data Preprocessing with Comprehensive Tools")
            processed_data, processed_targets = self.preprocess_data_with_comprehensive_tools(data, targets)
            
            # 2. Feature engineering with comprehensive tools
            tprint_info("🔧 Step 2: Feature Engineering with Comprehensive Tools")
            engineered_data = self._engineer_analyst_features_with_comprehensive_tools(processed_data)
            
            # 3. Model training with comprehensive tools
            tprint_info("🤖 Step 3: Model Training with Comprehensive Tools")
            trained_models = await self._train_analyst_models_with_comprehensive_tools(engineered_data, processed_targets)
            
            # 4. Model validation with comprehensive tools
            tprint_info("✅ Step 4: Model Validation with Comprehensive Tools")
            validation_metrics = await self._validate_analyst_models_with_comprehensive_tools(engineered_data, processed_targets, trained_models)
            
            # 5. Feature importance extraction with comprehensive tools
            tprint_info("📈 Step 5: Feature Importance Extraction with Comprehensive Tools")
            feature_importance = self._extract_feature_importance_with_comprehensive_tools(trained_models, engineered_data)
            
            # 6. Model saving with comprehensive tools
            tprint_info("💾 Step 6: Model Saving with Comprehensive Tools")
            saved_paths = self._save_analyst_models_with_comprehensive_tools(trained_models, validation_metrics, feature_importance)
            
            # 7. Performance monitoring with comprehensive tools
            tprint_info("📊 Step 7: Performance Monitoring with Comprehensive Tools")
            self._log_comprehensive_training_summary()
            
            # Create enhanced result
            result = ModelTrainingResult(
                success=True,
                models=trained_models,
                metrics=validation_metrics,
                training_time=time.time() - self._performance_start_time,
                validation_metrics=validation_metrics,
                feature_importance=feature_importance,
                artifacts=list(saved_paths.keys()),
                metadata={
                    'comprehensive_tools_used': True,
                    'enhanced_training': True,
                    'data_shape': processed_data.shape,
                    'features_engineered': len(engineered_data.columns),
                    'models_trained': len(trained_models),
                    'analyst_config': self.enhanced_config.__dict__
                }
            )
            
            tprint_operation_end("Enhanced Analyst Model Training", success=True)
            tprint_success("✅ Enhanced Analyst model training completed successfully")
            
            return result
            
        except Exception as e:
            tprint_operation_end("Enhanced Analyst Model Training", success=False)
            tprint_error(f"❌ Enhanced Analyst model training failed: {e}")
            return ModelTrainingResult(
                success=False,
                errors=[str(e)],
                metadata={'comprehensive_tools_used': True, 'enhanced_training': True}
            )
    
    @with_memory_optimization(level="AGGRESSIVE")
    def _engineer_analyst_features_with_comprehensive_tools(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer Analyst-specific features using comprehensive tools."""
        try:
            tprint_operation_start("Analyst Feature Engineering with Comprehensive Tools")
            
            # Use BaseStep data preview
            tprint_data_preview(data, "Input data for Analyst feature engineering", max_rows=5)
            
            # Create engineered features
            engineered_data = data.copy()
            
            # Price-based features using BaseStep safe operations
            if 'close' in data.columns:
                # Moving averages
                for window in [5, 10, 20, 50]:
                    engineered_data[f'sma_{window}'] = self._safe_dataframe_operation(
                        data, 'rolling', window=window
                    )['close'].mean()
                
                # Exponential moving averages
                for window in [12, 26]:
                    engineered_data[f'ema_{window}'] = self._safe_dataframe_operation(
                        data, 'ewm', span=window
                    )['close'].mean()
                
                # Technical indicators
                engineered_data = self._add_technical_indicators_with_comprehensive_tools(engineered_data)
            
            # Volume features if available
            if 'volume' in data.columns:
                engineered_data = self._add_volume_features_with_comprehensive_tools(engineered_data)
            
            # Regime features if enabled
            if self.enhanced_config.enable_regime_features:
                engineered_data = self._add_regime_features_with_comprehensive_tools(engineered_data)
            
            # Multi-timeframe features if enabled
            if self.enhanced_config.enable_multi_timeframe:
                engineered_data = self._add_multi_timeframe_features_with_comprehensive_tools(engineered_data)
            
            # Use BaseStep data validation
            engineered_data = self.comprehensive_tools.process_data_with_comprehensive_tools(
                engineered_data, "validate", required_columns=['close']
            )
            
            # Use BaseStep data preview for output
            tprint_data_preview(engineered_data, "Engineered Analyst features", max_rows=5)
            tprint_dataframe_info(engineered_data, "Engineered Analyst Data Info")
            
            tprint_operation_end("Analyst Feature Engineering with Comprehensive Tools", success=True)
            tprint_success(f"✅ Engineered {len(engineered_data.columns)} Analyst features")
            
            return engineered_data
            
        except Exception as e:
            tprint_operation_end("Analyst Feature Engineering with Comprehensive Tools", success=False)
            tprint_error(f"❌ Analyst feature engineering failed: {e}")
            raise
    
    def _add_technical_indicators_with_comprehensive_tools(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using comprehensive tools."""
        try:
            # RSI
            if 'close' in data.columns:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                data['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            if 'close' in data.columns and 'sma_20' in data.columns:
                sma_20 = data['sma_20']
                std_20 = self._safe_dataframe_operation(
                    data, 'rolling', window=20
                )['close'].std()
                data['bb_upper'] = sma_20 + (std_20 * 2)
                data['bb_lower'] = sma_20 - (std_20 * 2)
                data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / sma_20
                data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # MACD
            if 'close' in data.columns:
                ema_12 = data['ema_12'] if 'ema_12' in data.columns else self._safe_dataframe_operation(
                    data, 'ewm', span=12
                )['close'].mean()
                ema_26 = data['ema_26'] if 'ema_26' in data.columns else self._safe_dataframe_operation(
                    data, 'ewm', span=26
                )['close'].mean()
                data['macd'] = ema_12 - ema_26
                data['macd_signal'] = self._safe_dataframe_operation(
                    data, 'ewm', span=9
                )['macd'].mean()
                data['macd_histogram'] = data['macd'] - data['macd_signal']
            
            return data
            
        except Exception as e:
            tprint_warning(f"⚠️ Technical indicators failed: {e}")
            return data
    
    def _add_volume_features_with_comprehensive_tools(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume features using comprehensive tools."""
        try:
            # Volume moving averages
            for window in [5, 10, 20]:
                data[f'volume_sma_{window}'] = self._safe_dataframe_operation(
                    data, 'rolling', window=window
                )['volume'].mean()
            
            # Volume ratios
            if 'volume_sma_20' in data.columns:
                data['volume_ratio'] = data['volume'] / data['volume_sma_20']
                data['volume_ratio_sma'] = self._safe_dataframe_operation(
                    data, 'rolling', window=5
                )['volume_ratio'].mean()
            
            return data
            
        except Exception as e:
            tprint_warning(f"⚠️ Volume features failed: {e}")
            return data
    
    def _add_regime_features_with_comprehensive_tools(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add regime features using comprehensive tools."""
        try:
            # Volatility regime
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                volatility = returns.rolling(window=20).std()
                data['volatility_regime'] = (volatility > volatility.rolling(window=50).mean()).astype(int)
            
            # Trend regime
            if 'sma_20' in data.columns and 'sma_50' in data.columns:
                data['trend_regime'] = (data['sma_20'] > data['sma_50']).astype(int)
            
            return data
            
        except Exception as e:
            tprint_warning(f"⚠️ Regime features failed: {e}")
            return data
    
    def _add_multi_timeframe_features_with_comprehensive_tools(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add multi-timeframe features using comprehensive tools."""
        try:
            # Higher timeframe features (simplified)
            if 'close' in data.columns:
                # Daily features
                data['daily_high'] = self._safe_dataframe_operation(
                    data, 'rolling', window=96  # Assuming 15m data
                )['close'].max()
                data['daily_low'] = self._safe_dataframe_operation(
                    data, 'rolling', window=96
                )['close'].min()
                data['daily_range'] = data['daily_high'] - data['daily_low']
                data['daily_position'] = (data['close'] - data['daily_low']) / data['daily_range']
            
            return data
            
        except Exception as e:
            tprint_warning(f"⚠️ Multi-timeframe features failed: {e}")
            return data
    
    @with_performance_tracking("Analyst Model Training with Comprehensive Tools")
    async def _train_analyst_models_with_comprehensive_tools(
        self, 
        data: pd.DataFrame, 
        targets: pd.Series
    ) -> Dict[str, Any]:
        """Train Analyst models using comprehensive tools."""
        try:
            tprint_operation_start("Analyst Model Training with Comprehensive Tools")
            
            trained_models = {}
            
            # Train LightGBM model
            if 'lightgbm' in self.enhanced_config.model_types:
                tprint_info("🤖 Training LightGBM model with comprehensive tools")
                lgb_model = await self._train_lightgbm_with_comprehensive_tools(data, targets)
                if lgb_model is not None:
                    trained_models['lightgbm'] = lgb_model
                    tprint_model_info(lgb_model, "LightGBM Model")
            
            # Train CatBoost model
            if 'catboost' in self.enhanced_config.model_types:
                tprint_info("🤖 Training CatBoost model with comprehensive tools")
                cat_model = await self._train_catboost_with_comprehensive_tools(data, targets)
                if cat_model is not None:
                    trained_models['catboost'] = cat_model
                    tprint_model_info(cat_model, "CatBoost Model")
            
            tprint_operation_end("Analyst Model Training with Comprehensive Tools", success=True)
            tprint_success(f"✅ Trained {len(trained_models)} Analyst models")
            
            return trained_models
            
        except Exception as e:
            tprint_operation_end("Analyst Model Training with Comprehensive Tools", success=False)
            tprint_error(f"❌ Analyst model training failed: {e}")
            raise
    
    async def _train_lightgbm_with_comprehensive_tools(
        self, 
        data: pd.DataFrame, 
        targets: pd.Series
    ) -> Any:
        """Train LightGBM model using comprehensive tools."""
        try:
            import lightgbm as lgb
            
            # Use BaseStep data validation
            if not self._validate_dataframe_columns(data, ['close']):
                raise ValueError("Required columns missing for LightGBM training")
            
            # Prepare data using BaseStep utilities
            X_train, X_val, y_train, y_val = self._split_data_with_comprehensive_tools(data, targets)
            
            # Create LightGBM dataset using BaseStep safe operations
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Default LightGBM parameters
            default_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0
            }
            
            # Merge with custom parameters
            params = {**default_params, **self.enhanced_config.lightgbm_params}
            
            # Train model with comprehensive monitoring
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Log model performance using BaseStep utilities
            tprint_performance(f"LightGBM training completed with {model.num_trees()} trees")
            
            return model
            
        except ImportError:
            tprint_warning("⚠️ LightGBM not available, skipping")
            return None
        except Exception as e:
            tprint_error(f"❌ LightGBM training failed: {e}")
            raise
    
    async def _train_catboost_with_comprehensive_tools(
        self, 
        data: pd.DataFrame, 
        targets: pd.Series
    ) -> Any:
        """Train CatBoost model using comprehensive tools."""
        try:
            from catboost import CatBoostRegressor
            
            # Use BaseStep data validation
            if not self._validate_dataframe_columns(data, ['close']):
                raise ValueError("Required columns missing for CatBoost training")
            
            # Prepare data using BaseStep utilities
            X_train, X_val, y_train, y_val = self._split_data_with_comprehensive_tools(data, targets)
            
            # Default CatBoost parameters
            default_params = {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'loss_function': 'RMSE',
                'verbose': False
            }
            
            # Merge with custom parameters
            params = {**default_params, **self.enhanced_config.catboost_params}
            
            # Train model with comprehensive monitoring
            model = CatBoostRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )
            
            # Log model performance using BaseStep utilities
            tprint_performance(f"CatBoost training completed with {model.get_best_iteration()} iterations")
            
            return model
            
        except ImportError:
            tprint_warning("⚠️ CatBoost not available, skipping")
            return None
        except Exception as e:
            tprint_error(f"❌ CatBoost training failed: {e}")
            raise
    
    def _split_data_with_comprehensive_tools(
        self, 
        data: pd.DataFrame, 
        targets: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data using comprehensive tools."""
        try:
            from sklearn.model_selection import train_test_split
            
            # Use BaseStep safe operations for data splitting
            X_train, X_val, y_train, y_val = train_test_split(
                data, targets,
                test_size=self.enhanced_config.validation_split,
                random_state=self.enhanced_config.random_seed
            )
            
            # Log split information using BaseStep utilities
            tprint_info(f"📊 Data split: Train={len(X_train)}, Val={len(X_val)}")
            
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            tprint_error(f"❌ Data splitting failed: {e}")
            raise
    
    async def _validate_analyst_models_with_comprehensive_tools(
        self, 
        data: pd.DataFrame, 
        targets: pd.Series, 
        models: Dict[str, Any]
    ) -> Dict[str, float]:
        """Validate Analyst models using comprehensive tools."""
        try:
            tprint_operation_start("Analyst Model Validation with Comprehensive Tools")
            
            validation_metrics = {}
            
            # Split data for validation
            X_train, X_val, y_train, y_val = self._split_data_with_comprehensive_tools(data, targets)
            
            for model_name, model in models.items():
                if model is None:
                    continue
                
                # Make predictions
                if hasattr(model, 'predict'):
                    predictions = model.predict(X_val)
                else:
                    predictions = model.predict(X_val)
                
                # Calculate metrics using BaseStep safe operations
                mse = np.mean((predictions - y_val) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predictions - y_val))
                
                # Calculate R² using BaseStep safe operations
                ss_res = np.sum((y_val - predictions) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                r2 = self._safe_divide(1 - ss_res, ss_tot, 0.0)
                
                validation_metrics[model_name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                # Log metrics using BaseStep utilities
                tprint_dict(validation_metrics[model_name], f"{model_name} Validation Metrics")
            
            tprint_operation_end("Analyst Model Validation with Comprehensive Tools", success=True)
            tprint_success("✅ Analyst model validation completed")
            
            return validation_metrics
            
        except Exception as e:
            tprint_operation_end("Analyst Model Validation with Comprehensive Tools", success=False)
            tprint_error(f"❌ Analyst model validation failed: {e}")
            raise
    
    def _extract_feature_importance_with_comprehensive_tools(
        self, 
        models: Dict[str, Any], 
        data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Extract feature importance using comprehensive tools."""
        try:
            tprint_operation_start("Feature Importance Extraction with Comprehensive Tools")
            
            feature_importance = {}
            
            for model_name, model in models.items():
                if model is None:
                    continue
                
                try:
                    if hasattr(model, 'feature_importance_'):
                        # LightGBM
                        importance = model.feature_importance_
                        feature_names = data.columns.tolist()
                    elif hasattr(model, 'get_feature_importance'):
                        # CatBoost
                        importance = model.get_feature_importance()
                        feature_names = data.columns.tolist()
                    else:
                        tprint_warning(f"⚠️ Feature importance not available for {model_name}")
                        continue
                    
                    # Create feature importance dictionary
                    feature_importance[model_name] = dict(zip(feature_names, importance))
                    
                    # Log feature importance using BaseStep utilities
                    tprint_dict(feature_importance[model_name], f"{model_name} Feature Importance")
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Feature importance extraction failed for {model_name}: {e}")
                    continue
            
            tprint_operation_end("Feature Importance Extraction with Comprehensive Tools", success=True)
            tprint_success("✅ Feature importance extraction completed")
            
            return feature_importance
            
        except Exception as e:
            tprint_operation_end("Feature Importance Extraction with Comprehensive Tools", success=False)
            tprint_error(f"❌ Feature importance extraction failed: {e}")
            return {}
    
    def _save_analyst_models_with_comprehensive_tools(
        self, 
        models: Dict[str, Any], 
        validation_metrics: Dict[str, float],
        feature_importance: Dict[str, Dict[str, float]]
    ) -> Dict[str, str]:
        """Save Analyst models using comprehensive tools."""
        try:
            tprint_operation_start("Analyst Model Saving with Comprehensive Tools")
            
            # Create comprehensive metadata
            metadata = {
                'validation_metrics': validation_metrics,
                'feature_importance': feature_importance,
                'enhanced_config': self.enhanced_config.__dict__,
                'comprehensive_tools_used': True,
                'timestamp': time.strftime("%Y%m%d_%H%M%S")
            }
            
            # Use BaseStep model saving utilities
            saved_paths = self.save_models_with_comprehensive_tools(models, metadata)
            
            tprint_operation_end("Analyst Model Saving with Comprehensive Tools", success=True)
            tprint_success(f"✅ Saved {len(models)} Analyst models")
            
            return saved_paths
            
        except Exception as e:
            tprint_operation_end("Analyst Model Saving with Comprehensive Tools", success=False)
            tprint_error(f"❌ Analyst model saving failed: {e}")
            raise
    
    # ============================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================================
    
    async def validate_models(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> Dict[str, float]:
        """Validate trained models."""
        # This would be implemented based on specific requirements
        return {}
    
    async def predict(self, data: pd.DataFrame, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Make predictions with trained models."""
        # This would be implemented based on specific requirements
        return {}
    
    def _create_model(self, model_type: ModelType) -> Any:
        """Create a model instance."""
        # This would be implemented based on specific model types
        return None