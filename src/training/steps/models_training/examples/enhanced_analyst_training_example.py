"""
Enhanced Analyst Training Example - Comprehensive Tools Integration

This example demonstrates how to use the generalized comprehensive tools
from BaseStep in a model training component.

Key Features Demonstrated:
- Using GeneralizedModelTrainingBase
- Comprehensive tools integration
- Advanced data processing
- Model management with comprehensive tools
- Performance monitoring
- Error handling and logging
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
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
    tprint_banner, tprint_separator, tprint_header, tprint_footer,
    tprint_operation_start, tprint_operation_end, tprint_data_preview,
    tprint_dict, tprint_list, tprint_dataframe_info, tprint_model_info,
    tprint_performance_summary, tprint_memory_usage, tprint_hardware_stats
)


class EnhancedAnalystTrainingExample(GeneralizedModelTrainingBase):
    """
    Enhanced Analyst Training Example using comprehensive tools.
    
    This example shows how to leverage all BaseStep comprehensive tools
    in a model training component.
    """
    
    def __init__(
        self,
        step_name: str = "enhanced_analyst_training_example",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the enhanced analyst training example.
        
        Args:
            step_name: Name of the training step
            config: Configuration dictionary
            logger: Logger instance (optional)
        """
        # Set up analyst-specific configuration
        if config is None:
            config = {}
        
        # Add analyst-specific defaults
        analyst_config = {
            'role': 'analyst',
            'model_types': ['lightgbm', 'catboost'],
            'timeframe': '15m',
            'symbol': 'ETHUSDT',
            'enable_patchtst_features': True,
            'enable_regime_features': True,
            'enable_multi_timeframe': True,
            'lightgbm_params': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0
            },
            'catboost_params': {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'loss_function': 'RMSE',
                'verbose': False
            }
        }
        
        # Merge with provided config
        analyst_config.update(config)
        
        super().__init__(step_name, analyst_config, logger)
        
        # Initialize comprehensive tools integration
        self.comprehensive_tools = ComprehensiveToolsIntegration(
            self, 
            ComprehensiveToolsConfig(
                enable_logging=True,
                enable_performance_monitoring=True,
                enable_memory_optimization=True,
                enable_hardware_optimization=True,
                enable_error_handling=True,
                log_level="INFO"
            )
        )
        
        tprint_banner("Enhanced Analyst Training Example")
        tprint_info("🔧 Initialized with comprehensive tools integration")
    
    # ============================================================================
    # COMPREHENSIVE TOOLS USAGE EXAMPLES
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
        Train models using comprehensive tools.
        
        This method demonstrates the full power of comprehensive tools integration.
        """
        try:
            tprint_operation_start("Enhanced Model Training with Comprehensive Tools")
            
            # 1. Data preprocessing with comprehensive tools
            tprint_info("📊 Step 1: Data Preprocessing with Comprehensive Tools")
            processed_data, processed_targets = self.preprocess_data_with_comprehensive_tools(data, targets)
            
            # 2. Feature engineering with comprehensive tools
            tprint_info("🔧 Step 2: Feature Engineering with Comprehensive Tools")
            engineered_data = self._engineer_features_with_comprehensive_tools(processed_data)
            
            # 3. Model training with comprehensive tools
            tprint_info("🤖 Step 3: Model Training with Comprehensive Tools")
            trained_models = await self._train_models_with_comprehensive_tools(engineered_data, processed_targets)
            
            # 4. Model validation with comprehensive tools
            tprint_info("✅ Step 4: Model Validation with Comprehensive Tools")
            validation_metrics = await self._validate_models_with_comprehensive_tools(engineered_data, processed_targets, trained_models)
            
            # 5. Model saving with comprehensive tools
            tprint_info("💾 Step 5: Model Saving with Comprehensive Tools")
            saved_paths = self._save_models_with_comprehensive_tools(trained_models, validation_metrics)
            
            # 6. Performance monitoring with comprehensive tools
            tprint_info("📈 Step 6: Performance Monitoring with Comprehensive Tools")
            self._log_comprehensive_training_summary()
            
            # Create result
            result = ModelTrainingResult(
                success=True,
                models=trained_models,
                metrics=validation_metrics,
                training_time=time.time() - self._performance_start_time,
                artifacts=list(saved_paths.keys()),
                metadata={
                    'comprehensive_tools_used': True,
                    'data_shape': processed_data.shape,
                    'features_engineered': len(engineered_data.columns),
                    'models_trained': len(trained_models)
                }
            )
            
            tprint_operation_end("Enhanced Model Training with Comprehensive Tools", success=True)
            tprint_success("✅ Enhanced model training completed successfully")
            
            return result
            
        except Exception as e:
            tprint_operation_end("Enhanced Model Training with Comprehensive Tools", success=False)
            tprint_error(f"❌ Enhanced model training failed: {e}")
            return ModelTrainingResult(
                success=False,
                errors=[str(e)],
                metadata={'comprehensive_tools_used': True}
            )
    
    @with_memory_optimization(level="AGGRESSIVE")
    def _engineer_features_with_comprehensive_tools(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features using comprehensive tools."""
        try:
            tprint_operation_start("Feature Engineering with Comprehensive Tools")
            
            # Use BaseStep data preview
            tprint_data_preview(data, "Input data for feature engineering", max_rows=5)
            
            # Create engineered features
            engineered_data = data.copy()
            
            # Technical indicators using BaseStep safe operations
            if 'close' in data.columns:
                # Moving averages
                engineered_data['sma_20'] = self._safe_dataframe_operation(
                    data, 'rolling', window=20
                )['close'].mean()
                engineered_data['sma_50'] = self._safe_dataframe_operation(
                    data, 'rolling', window=50
                )['close'].mean()
                
                # RSI using BaseStep safe operations
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                engineered_data['rsi'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                sma_20 = engineered_data['sma_20']
                std_20 = self._safe_dataframe_operation(
                    data, 'rolling', window=20
                )['close'].std()
                engineered_data['bb_upper'] = sma_20 + (std_20 * 2)
                engineered_data['bb_lower'] = sma_20 - (std_20 * 2)
                engineered_data['bb_width'] = (engineered_data['bb_upper'] - engineered_data['bb_lower']) / sma_20
            
            # Volume features if available
            if 'volume' in data.columns:
                engineered_data['volume_sma_20'] = self._safe_dataframe_operation(
                    data, 'rolling', window=20
                )['volume'].mean()
                engineered_data['volume_ratio'] = data['volume'] / engineered_data['volume_sma_20']
            
            # Price features
            if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
                engineered_data['price_range'] = data['high'] - data['low']
                engineered_data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
            
            # Use BaseStep data validation
            engineered_data = self.comprehensive_tools.process_data_with_comprehensive_tools(
                engineered_data, "validate", required_columns=['close']
            )
            
            # Use BaseStep data preview for output
            tprint_data_preview(engineered_data, "Engineered features", max_rows=5)
            tprint_dataframe_info(engineered_data, "Engineered Data Info")
            
            tprint_operation_end("Feature Engineering with Comprehensive Tools", success=True)
            tprint_success(f"✅ Engineered {len(engineered_data.columns)} features")
            
            return engineered_data
            
        except Exception as e:
            tprint_operation_end("Feature Engineering with Comprehensive Tools", success=False)
            tprint_error(f"❌ Feature engineering failed: {e}")
            raise
    
    @with_performance_tracking("Model Training with Comprehensive Tools")
    async def _train_models_with_comprehensive_tools(
        self, 
        data: pd.DataFrame, 
        targets: pd.Series
    ) -> Dict[str, Any]:
        """Train models using comprehensive tools."""
        try:
            tprint_operation_start("Model Training with Comprehensive Tools")
            
            trained_models = {}
            
            # Train LightGBM model
            if ModelType.LIGHTGBM in self.training_config.model_types:
                tprint_info("🤖 Training LightGBM model with comprehensive tools")
                lgb_model = await self._train_lightgbm_with_comprehensive_tools(data, targets)
                trained_models['lightgbm'] = lgb_model
                tprint_model_info(lgb_model, "LightGBM Model")
            
            # Train CatBoost model
            if ModelType.CATBOOST in self.training_config.model_types:
                tprint_info("🤖 Training CatBoost model with comprehensive tools")
                cat_model = await self._train_catboost_with_comprehensive_tools(data, targets)
                trained_models['catboost'] = cat_model
                tprint_model_info(cat_model, "CatBoost Model")
            
            tprint_operation_end("Model Training with Comprehensive Tools", success=True)
            tprint_success(f"✅ Trained {len(trained_models)} models")
            
            return trained_models
            
        except Exception as e:
            tprint_operation_end("Model Training with Comprehensive Tools", success=False)
            tprint_error(f"❌ Model training failed: {e}")
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
            
            # Train model with comprehensive monitoring
            model = lgb.train(
                self.training_config.custom_params.get('lightgbm_params', {}),
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Log model performance using BaseStep utilities
            tprint_performance(f"LightGBM training completed with {model.num_trees()} trees")
            
            return model
            
        except ImportError:
            tprint_warning("⚠️ LightGBM not available, using fallback")
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
            
            # Train model with comprehensive monitoring
            model = CatBoostRegressor(
                **self.training_config.custom_params.get('catboost_params', {}),
                random_seed=self.training_config.random_seed
            )
            
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )
            
            # Log model performance using BaseStep utilities
            tprint_performance(f"CatBoost training completed with {model.get_best_iteration()} iterations")
            
            return model
            
        except ImportError:
            tprint_warning("⚠️ CatBoost not available, using fallback")
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
                test_size=self.training_config.validation_split,
                random_seed=self.training_config.random_seed
            )
            
            # Log split information using BaseStep utilities
            tprint_info(f"📊 Data split: Train={len(X_train)}, Val={len(X_val)}")
            
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            tprint_error(f"❌ Data splitting failed: {e}")
            raise
    
    async def _validate_models_with_comprehensive_tools(
        self, 
        data: pd.DataFrame, 
        targets: pd.Series, 
        models: Dict[str, Any]
    ) -> Dict[str, float]:
        """Validate models using comprehensive tools."""
        try:
            tprint_operation_start("Model Validation with Comprehensive Tools")
            
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
            
            tprint_operation_end("Model Validation with Comprehensive Tools", success=True)
            tprint_success("✅ Model validation completed")
            
            return validation_metrics
            
        except Exception as e:
            tprint_operation_end("Model Validation with Comprehensive Tools", success=False)
            tprint_error(f"❌ Model validation failed: {e}")
            raise
    
    def _save_models_with_comprehensive_tools(
        self, 
        models: Dict[str, Any], 
        validation_metrics: Dict[str, float]
    ) -> Dict[str, str]:
        """Save models using comprehensive tools."""
        try:
            tprint_operation_start("Model Saving with Comprehensive Tools")
            
            # Create comprehensive metadata
            metadata = {
                'validation_metrics': validation_metrics,
                'training_config': self.training_config.__dict__,
                'comprehensive_tools_used': True,
                'timestamp': time.strftime("%Y%m%d_%H%M%S")
            }
            
            # Use BaseStep model saving utilities
            saved_paths = self.save_models_with_comprehensive_tools(models, metadata)
            
            tprint_operation_end("Model Saving with Comprehensive Tools", success=True)
            tprint_success(f"✅ Saved {len(models)} models")
            
            return saved_paths
            
        except Exception as e:
            tprint_operation_end("Model Saving with Comprehensive Tools", success=False)
            tprint_error(f"❌ Model saving failed: {e}")
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


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def run_enhanced_analyst_training_example():
    """Run the enhanced analyst training example."""
    try:
        tprint_banner("Enhanced Analyst Training Example")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame({
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 105,
            'low': np.random.randn(n_samples).cumsum() + 95,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        # Create targets
        targets = data['close'].pct_change().shift(-1).dropna()
        data = data.iloc[:-1]  # Remove last row to match targets
        
        # Initialize enhanced analyst training
        trainer = EnhancedAnalystTrainingExample()
        
        # Train models with comprehensive tools
        result = await trainer.train_models(data, targets)
        
        # Print results
        tprint_dict(result.__dict__, "Training Result")
        
        # Print comprehensive tools status
        trainer.print_comprehensive_tools_help()
        
        tprint_success("✅ Enhanced analyst training example completed")
        
    except Exception as e:
        tprint_error(f"❌ Enhanced analyst training example failed: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_enhanced_analyst_training_example())