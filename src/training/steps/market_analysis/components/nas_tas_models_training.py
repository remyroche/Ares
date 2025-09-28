"""
NAS-TAS Models Training Component

This component implements base model training for NAS-TAS (Neural Architecture Search - Tree-based Architecture Search) based regime detection models.
It trains individual base models using NAS-TAS regime labels for regime classification.
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from src.utils.logger import system_logger
from src.utils.tprint import tprint
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Suppress LightGBM warnings about no further splits
warnings.filterwarnings('ignore', message='.*No further splits with positive gain.*')

# Import ML libraries with error handling
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    import lightgbm as lgb
    ML_LIBRARIES_AVAILABLE = True
    tprint("✅ [NAS_TAS_MODELS] ML libraries imported successfully", color="green")
except ImportError as e:
    ML_LIBRARIES_AVAILABLE = False
    tprint(f"❌ [NAS_TAS_MODELS] Failed to import ML libraries: {e}", color="red")


class NASTASModelsTrainingComponent(BaseMarketAnalysisComponent):
    """
    NAS-TAS Models Training Component.
    
    This component trains base models using NAS-TAS regime labels for regime classification.
    It creates individual models for regime detection and classification.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the NAS-TAS Models Training Component."""
        tprint("🚀 [NAS_TAS_MODELS] Initializing NAS-TAS Models Training Component", color="cyan", bold=True)
        super().__init__(config)
        
        self.logger = system_logger.getChild('NASTASModelsTrainingComponent')
        tprint("✅ [NAS_TAS_MODELS] Logger initialized", color="green")
        
        # Initialize model training parameters
        self.model_config = {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'n_jobs': -1
        }
        
        # Initialize models
        self.models = {}
        self.model_metrics = {}
        tprint("📊 [NAS_TAS_MODELS] Models initialized", color="blue")
        
        tprint("✅ [NAS_TAS_MODELS] NAS-TAS Models Training Component initialized successfully", color="green", bold=True)
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint("📋 [NAS_TAS_MODELS] Getting required artifacts", color="cyan")
        required_artifacts = ['nas_tas_models_training_result']
        tprint(f"✅ [NAS_TAS_MODELS] Required artifacts: {required_artifacts}", color="green")
        return required_artifacts
    
    async def execute(self, data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute NAS-TAS models training.
        
        Args:
            data: Market data DataFrame
            pipeline_state: Pipeline state dictionary
            
        Returns:
            ComponentResult with training results
        """
        tprint("🚀 [NAS_TAS_MODELS] Starting NAS-TAS models training execution", color="cyan", bold=True)
        
        try:
            # Check if ML libraries are available
            if not ML_LIBRARIES_AVAILABLE:
                error_msg = "ML libraries not available for NAS-TAS models training"
                tprint(f"❌ [NAS_TAS_MODELS] {error_msg}", color="red")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            
            # Extract regime labels from pipeline state artifacts
            artifacts = pipeline_state.get('artifacts', {})
            nas_tas_clustering_result = artifacts.get('nas_tas_clustering_result', {})
            regime_labels = nas_tas_clustering_result.get('regime_assignments')
            
            if regime_labels is None:
                error_msg = "No regime labels found in pipeline state artifacts"
                tprint(f"❌ [NAS_TAS_MODELS] {error_msg}", color="red")
                tprint(f"🔍 [NAS_TAS_MODELS] Available artifacts: {list(artifacts.keys())}", color="yellow")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            
            tprint(f"📊 [NAS_TAS_MODELS] Found regime labels: {len(regime_labels)} samples", color="blue")
            
            # Prepare features and targets
            X, y = self._prepare_training_data(data, regime_labels)
            if X is None or y is None:
                error_msg = "Failed to prepare training data"
                tprint(f"❌ [NAS_TAS_MODELS] {error_msg}", color="red")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            
            tprint(f"📊 [NAS_TAS_MODELS] Training data prepared: X={X.shape}, y={y.shape}", color="blue")
            
            # Train models
            training_results = self._train_models(X, y)
            
            # Create artifacts
            artifacts = {
                'nas_tas_models_training_result': {
                    'models': training_results['models'],
                    'metrics': training_results['metrics'],
                    'training_time': training_results['training_time'],
                    'success': True
                }
            }
            
            tprint("✅ [NAS_TAS_MODELS] NAS-TAS models training completed successfully", color="green", bold=True)
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={'component_type': 'nas_tas_models_training', 'execution_time': training_results['training_time']}
            )
            
        except Exception as e:
            error_msg = f"NAS-TAS models training failed: {str(e)}"
            tprint(f"❌ [NAS_TAS_MODELS] {error_msg}", color="red")
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=error_msg
            )
    
    def _prepare_training_data(self, data: pd.DataFrame, regime_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from market data and regime labels."""
        tprint("🔧 [NAS_TAS_MODELS] Preparing training data", color="cyan")
        
        try:
            # Create basic features from OHLCV data
            features = []
            
            if 'close' in data.columns:
                # Price-based features
                returns = data['close'].pct_change().fillna(0)
                features.append(returns.values)
                
                # Moving averages
                sma_20 = data['close'].rolling(20).mean().fillna(data['close'].iloc[0])
                sma_50 = data['close'].rolling(50).mean().fillna(data['close'].iloc[0])
                features.append(sma_20.values)
                features.append(sma_50.values)
                
                # Volatility
                volatility = returns.rolling(20).std().fillna(0)
                features.append(volatility.values)
            
            if 'volume' in data.columns:
                # Volume features
                volume_ratio = data['volume'] / data['volume'].rolling(20).mean().fillna(data['volume'].mean())
                features.append(volume_ratio.fillna(1).values)
            
            # Combine features
            X = np.column_stack(features)
            
            # Align with regime labels
            min_length = min(len(X), len(regime_labels))
            X = X[:min_length]
            y = np.array(regime_labels[:min_length])
            
            tprint(f"✅ [NAS_TAS_MODELS] Training data prepared: {X.shape[0]} samples, {X.shape[1]} features", color="green")
            return X, y
            
        except Exception as e:
            tprint(f"❌ [NAS_TAS_MODELS] Error preparing training data: {e}", color="red")
            return None, None
    
    def _train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train individual models for regime classification."""
        tprint("🏋️ [NAS_TAS_MODELS] Training models", color="cyan")
        
        start_time = time.time()
        models = {}
        metrics = {}
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.model_config['test_size'], 
                random_state=self.model_config['random_state'], 
                stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            tprint("🌲 [NAS_TAS_MODELS] Training Random Forest", color="blue")
            rf_model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.model_config['random_state'],
                n_jobs=self.model_config['n_jobs']
            )
            rf_model.fit(X_train_scaled, y_train)
            models['random_forest'] = rf_model
            
            # Train Logistic Regression
            tprint("📊 [NAS_TAS_MODELS] Training Logistic Regression", color="blue")
            lr_model = LogisticRegression(
                random_state=self.model_config['random_state'],
                max_iter=1000
            )
            lr_model.fit(X_train_scaled, y_train)
            models['logistic_regression'] = lr_model
            
            # Train XGBoost
            tprint("🚀 [NAS_TAS_MODELS] Training XGBoost", color="blue")
            xgb_model = xgb.XGBClassifier(
                random_state=self.model_config['random_state'],
                n_jobs=self.model_config['n_jobs']
            )
            xgb_model.fit(X_train_scaled, y_train)
            models['xgboost'] = xgb_model
            
            # Train LightGBM
            tprint("💡 [NAS_TAS_MODELS] Training LightGBM", color="blue")
            lgb_model = lgb.LGBMClassifier(
                random_state=self.model_config['random_state'],
                n_jobs=self.model_config['n_jobs']
            )
            lgb_model.fit(X_train_scaled, y_train)
            models['lightgbm'] = lgb_model
            
            # Evaluate models
            for name, model in models.items():
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                metrics[name] = {
                    'accuracy': accuracy,
                    'test_samples': len(y_test)
                }
                tprint(f"📊 [NAS_TAS_MODELS] {name} accuracy: {accuracy:.4f}", color="green")
            
            training_time = time.time() - start_time
            tprint(f"⏱️ [NAS_TAS_MODELS] Training completed in {training_time:.2f} seconds", color="blue")
            
            return {
                'models': models,
                'metrics': metrics,
                'training_time': training_time,
                'scaler': scaler
            }
            
        except Exception as e:
            tprint(f"❌ [NAS_TAS_MODELS] Error training models: {e}", color="red")
            return {
                'models': {},
                'metrics': {},
                'training_time': 0
            }
