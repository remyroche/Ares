"""
Enhanced CVLSA Integration with Tree Models

This module provides integration between the enhanced CVLSA architecture
and existing tree-based models, combining the benefits of both approaches.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

# Import existing components
from .enhanced_cvlsa_architecture import (
    EnhancedCVLSAConfig, EnhancedCVLSATrainer, create_enhanced_cvlsa_model
)
from .tree_clvsa_wrapper import TreeCLVSAWrapper, TreeCLVSAConfig
from src.utils.matrix_operations.enhanced_operations import get_enhanced_matrix_operations
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer

logger = logging.getLogger(__name__)


class HybridCVLSATreeModel(BaseEstimator, RegressorMixin):
    """
    Hybrid model combining enhanced CVLSA with tree-based models.
    
    This model uses CVLSA for feature extraction and attention mechanisms,
    then feeds the enhanced features to tree-based models for final prediction.
    """
    
    def __init__(self, 
                 cvlsa_config: Optional[EnhancedCVLSAConfig] = None,
                 tree_config: Optional[TreeCLVSAConfig] = None,
                 tree_model_type: str = 'random_forest',
                 fusion_method: str = 'weighted_average',
                 cvlsa_weight: float = 0.6,
                 tree_weight: float = 0.4):
        """
        Initialize hybrid CVLSA-tree model.
        
        Args:
            cvlsa_config: CVLSA configuration
            tree_config: Tree model configuration
            tree_model_type: Type of tree model ('random_forest', 'extra_trees', 'xgboost', 'lightgbm', 'catboost')
            fusion_method: Method to fuse CVLSA and tree predictions ('weighted_average', 'stacking', 'attention')
            cvlsa_weight: Weight for CVLSA predictions in fusion
            tree_weight: Weight for tree predictions in fusion
        """
        self.cvlsa_config = cvlsa_config or EnhancedCVLSAConfig()
        self.tree_config = tree_config or TreeCLVSAConfig()
        self.tree_model_type = tree_model_type
        self.fusion_method = fusion_method
        self.cvlsa_weight = cvlsa_weight
        self.tree_weight = tree_weight
        
        # Initialize components
        self.cvlsa_model = None
        self.tree_model = None
        self.fusion_model = None
        self.feature_scaler = StandardScaler()
        
        # Initialize hardware optimizers
        self._init_hardware_optimizers()
        
        # Training state
        self.is_fitted = False
        self.training_metadata = {}
        
        logger.info(f"🌳 Hybrid CVLSA-Tree model initialized (fusion: {fusion_method})")
    
    def _init_hardware_optimizers(self):
        """Initialize hardware optimization components."""
        try:
            self.matrix_ops = get_enhanced_matrix_operations()
            self.gpu_manager = get_m1_gpu_manager() if self.cvlsa_config.use_m1_gpu else None
            self.memory_optimizer = get_m1_memory_optimizer(
                memory_limit_gb=self.cvlsa_config.memory_limit_gb
            )
        except Exception as e:
            logger.warning(f"Hardware optimizers not available: {e}")
            self.matrix_ops = None
            self.gpu_manager = None
            self.memory_optimizer = None
    
    def _create_tree_model(self) -> BaseEstimator:
        """Create tree model based on specified type."""
        if self.tree_model_type == 'random_forest':
            base_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.tree_model_type == 'extra_trees':
            base_model = ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.tree_model_type == 'xgboost':
            try:
                import xgboost as xgb
                base_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            except ImportError:
                logger.warning("XGBoost not available, falling back to RandomForest")
                base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.tree_model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                base_model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
            except ImportError:
                logger.warning("LightGBM not available, falling back to RandomForest")
                base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.tree_model_type == 'catboost':
            try:
                from catboost import CatBoostRegressor
                base_model = CatBoostRegressor(
                    iterations=100,
                    depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=False
                )
            except ImportError:
                logger.warning("CatBoost not available, falling back to RandomForest")
                base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported tree model type: {self.tree_model_type}")
        
        # Wrap with TreeCLVSA if configured
        if self.tree_config:
            return TreeCLVSAWrapper(base_model, self.tree_config)
        else:
            return base_model
    
    def _create_fusion_model(self) -> BaseEstimator:
        """Create fusion model for combining CVLSA and tree predictions."""
        if self.fusion_method == 'weighted_average':
            return None  # Simple weighted average, no model needed
        elif self.fusion_method == 'stacking':
            # Use a simple linear model for stacking
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        elif self.fusion_method == 'attention':
            # Use a neural network for attention-based fusion
            return self._create_attention_fusion_model()
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
    
    def _create_attention_fusion_model(self) -> nn.Module:
        """Create attention-based fusion model."""
        class AttentionFusion(nn.Module):
            def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
                super().__init__()
                self.attention = nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=2,
                    batch_first=True
                )
                self.fusion = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
            
            def forward(self, cvlsa_pred: torch.Tensor, tree_pred: torch.Tensor) -> torch.Tensor:
                # Combine predictions
                combined = torch.stack([cvlsa_pred, tree_pred], dim=-1)
                
                # Apply attention
                attn_output, _ = self.attention(combined, combined, combined)
                
                # Fuse with attention
                fused = self.fusion(attn_output)
                
                return fused.squeeze(-1)
        
        return AttentionFusion()
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            market_data: Optional[pd.DataFrame] = None,
            regimes: Optional[np.ndarray] = None) -> 'HybridCVLSATreeModel':
        """
        Fit the hybrid model.
        
        Args:
            X: Input features
            y: Target values
            market_data: Market data for CVLSA feature preparation
            regimes: Regime labels for regime-aware training
            
        Returns:
            Self for method chaining
        """
        logger.info("🚀 Training hybrid CVLSA-tree model...")
        
        start_time = time.time()
        
        try:
            # Prepare market data for CVLSA
            if market_data is None:
                # Create synthetic market data from features
                market_data = self._create_synthetic_market_data(X)
            
            # Initialize CVLSA model
            self.cvlsa_model = create_enhanced_cvlsa_model(self.cvlsa_config)
            
            # Prepare CVLSA features
            cvlsa_features = self.cvlsa_model.prepare_features(market_data)
            
            # Train CVLSA model
            logger.info("🔧 Training CVLSA component...")
            cvlsa_results = self.cvlsa_model.train(
                cvlsa_features, cvlsa_features, torch.FloatTensor(y)
            )
            
            # Extract CVLSA predictions for tree training
            with torch.no_grad():
                cvlsa_predictions = self.cvlsa_model.predict(cvlsa_features)
                cvlsa_features_np = cvlsa_predictions.cpu().numpy()
            
            # Create enhanced features for tree model
            enhanced_features = np.hstack([X, cvlsa_features_np])
            
            # Scale features
            enhanced_features_scaled = self.feature_scaler.fit_transform(enhanced_features)
            
            # Initialize and train tree model
            logger.info("🌳 Training tree component...")
            self.tree_model = self._create_tree_model()
            
            if regimes is not None:
                self.tree_model.fit(enhanced_features_scaled, y, regimes=regimes)
            else:
                self.tree_model.fit(enhanced_features_scaled, y)
            
            # Initialize fusion model if needed
            if self.fusion_method in ['stacking', 'attention']:
                self.fusion_model = self._create_fusion_model()
                
                # Prepare fusion training data
                cvlsa_pred = self.cvlsa_model.predict(cvlsa_features).cpu().numpy()
                tree_pred = self.tree_model.predict(enhanced_features_scaled)
                
                fusion_X = np.column_stack([cvlsa_pred, tree_pred])
                
                if self.fusion_method == 'stacking':
                    self.fusion_model.fit(fusion_X, y)
                elif self.fusion_method == 'attention':
                    # Train attention fusion model
                    self._train_attention_fusion(fusion_X, y)
            
            # Store training metadata
            self.training_metadata = {
                'training_time': time.time() - start_time,
                'cvlsa_results': cvlsa_results,
                'feature_dimensions': {
                    'original': X.shape[1],
                    'cvlsa_enhanced': cvlsa_features_np.shape[1],
                    'total_enhanced': enhanced_features.shape[1]
                },
                'fusion_method': self.fusion_method,
                'tree_model_type': self.tree_model_type
            }
            
            self.is_fitted = True
            logger.info(f"✅ Hybrid model training completed in {self.training_metadata['training_time']:.2f}s")
            
            return self
            
        except Exception as e:
            logger.error(f"❌ Hybrid model training failed: {e}")
            raise
    
    def _create_synthetic_market_data(self, X: np.ndarray) -> pd.DataFrame:
        """Create synthetic market data from features for CVLSA."""
        n_samples = X.shape[0]
        
        # Create synthetic OHLCV data
        base_price = 100.0
        prices = []
        volumes = []
        
        for i in range(n_samples):
            # Generate price movement
            if i == 0:
                price = base_price
            else:
                # Use feature values to influence price movement
                feature_influence = np.mean(X[i, :min(5, X.shape[1])])  # Use first 5 features
                price_change = np.random.normal(0, 0.02) + feature_influence * 0.01
                price = prices[-1] * (1 + price_change)
            
            # Generate OHLC from price
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price * (1 + np.random.normal(0, 0.005))
            close = price
            
            prices.append([open_price, high, low, close])
            
            # Generate volume
            volume = np.random.lognormal(10, 1)
            volumes.append(volume)
        
        # Create DataFrame
        market_data = pd.DataFrame({
            'open': [p[0] for p in prices],
            'high': [p[1] for p in prices],
            'low': [p[2] for p in prices],
            'close': [p[3] for p in prices],
            'volume': volumes
        })
        
        return market_data
    
    def _train_attention_fusion(self, fusion_X: np.ndarray, y: np.ndarray):
        """Train attention-based fusion model."""
        # Convert to tensors
        fusion_X_tensor = torch.FloatTensor(fusion_X)
        y_tensor = torch.FloatTensor(y)
        
        # Training loop for attention fusion
        optimizer = torch.optim.Adam(self.fusion_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        for epoch in range(50):  # Simple training loop
            optimizer.zero_grad()
            
            # Split predictions
            cvlsa_pred = fusion_X_tensor[:, 0:1]
            tree_pred = fusion_X_tensor[:, 1:2]
            
            # Forward pass
            output = self.fusion_model(cvlsa_pred, tree_pred)
            loss = criterion(output, y_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
    
    def predict(self, X: np.ndarray, market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Make predictions with the hybrid model.
        
        Args:
            X: Input features
            market_data: Market data for CVLSA feature preparation
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Prepare market data for CVLSA
            if market_data is None:
                market_data = self._create_synthetic_market_data(X)
            
            # Get CVLSA predictions
            cvlsa_features = self.cvlsa_model.prepare_features(market_data)
            with torch.no_grad():
                cvlsa_predictions = self.cvlsa_model.predict(cvlsa_features)
                cvlsa_features_np = cvlsa_predictions.cpu().numpy()
            
            # Create enhanced features for tree model
            enhanced_features = np.hstack([X, cvlsa_features_np])
            enhanced_features_scaled = self.feature_scaler.transform(enhanced_features)
            
            # Get tree predictions
            tree_predictions = self.tree_model.predict(enhanced_features_scaled)
            
            # Fuse predictions
            if self.fusion_method == 'weighted_average':
                # Simple weighted average
                predictions = (self.cvlsa_weight * cvlsa_predictions.cpu().numpy() + 
                             self.tree_weight * tree_predictions)
            elif self.fusion_method == 'stacking':
                # Use fusion model
                fusion_X = np.column_stack([cvlsa_predictions.cpu().numpy(), tree_predictions])
                predictions = self.fusion_model.predict(fusion_X)
            elif self.fusion_method == 'attention':
                # Use attention fusion
                with torch.no_grad():
                    cvlsa_tensor = torch.FloatTensor(cvlsa_predictions.cpu().numpy()).unsqueeze(-1)
                    tree_tensor = torch.FloatTensor(tree_predictions).unsqueeze(-1)
                    predictions = self.fusion_model(cvlsa_tensor, tree_tensor).numpy()
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from both components."""
        importance = {}
        
        # CVLSA attention weights
        if hasattr(self.cvlsa_model, 'get_attention_weights'):
            importance['cvlsa_attention'] = self.cvlsa_model.get_attention_weights()
        
        # Tree feature importance
        if hasattr(self.tree_model, 'get_feature_importance'):
            importance['tree_importance'] = self.tree_model.get_feature_importance()
        elif hasattr(self.tree_model, 'base_model') and hasattr(self.tree_model.base_model, 'feature_importances_'):
            importance['tree_importance'] = self.tree_model.base_model.feature_importances_
        
        return importance
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_type': 'HybridCVLSATree',
            'cvlsa_config': self.cvlsa_config.__dict__,
            'tree_config': self.tree_config.__dict__,
            'tree_model_type': self.tree_model_type,
            'fusion_method': self.fusion_method,
            'is_fitted': self.is_fitted,
            'training_metadata': self.training_metadata
        }


class CVLSAFeatureExtractor:
    """
    Feature extractor that uses CVLSA to enhance features for any downstream model.
    """
    
    def __init__(self, cvlsa_config: Optional[EnhancedCVLSAConfig] = None):
        self.cvlsa_config = cvlsa_config or EnhancedCVLSAConfig()
        self.cvlsa_model = None
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info("🔧 CVLSA Feature Extractor initialized")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            market_data: Optional[pd.DataFrame] = None) -> 'CVLSAFeatureExtractor':
        """Fit the CVLSA feature extractor."""
        logger.info("🔧 Training CVLSA feature extractor...")
        
        # Prepare market data
        if market_data is None:
            market_data = self._create_synthetic_market_data(X)
        
        # Initialize and train CVLSA model
        self.cvlsa_model = create_enhanced_cvlsa_model(self.cvlsa_config)
        cvlsa_features = self.cvlsa_model.prepare_features(market_data)
        
        # Train CVLSA model
        cvlsa_results = self.cvlsa_model.train(
            cvlsa_features, cvlsa_features, torch.FloatTensor(y)
        )
        
        self.is_fitted = True
        logger.info("✅ CVLSA feature extractor trained")
        
        return self
    
    def transform(self, X: np.ndarray, market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Extract enhanced features using CVLSA."""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transforming")
        
        # Prepare market data
        if market_data is None:
            market_data = self._create_synthetic_market_data(X)
        
        # Get CVLSA features
        cvlsa_features = self.cvlsa_model.prepare_features(market_data)
        
        with torch.no_grad():
            cvlsa_predictions = self.cvlsa_model.predict(cvlsa_features)
            cvlsa_features_np = cvlsa_predictions.cpu().numpy()
        
        # Combine with original features
        enhanced_features = np.hstack([X, cvlsa_features_np])
        
        # Scale features
        enhanced_features_scaled = self.feature_scaler.fit_transform(enhanced_features)
        
        return enhanced_features_scaled
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, 
                     market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y, market_data).transform(X, market_data)
    
    def _create_synthetic_market_data(self, X: np.ndarray) -> pd.DataFrame:
        """Create synthetic market data from features."""
        n_samples = X.shape[0]
        base_price = 100.0
        prices = []
        volumes = []
        
        for i in range(n_samples):
            if i == 0:
                price = base_price
            else:
                feature_influence = np.mean(X[i, :min(5, X.shape[1])])
                price_change = np.random.normal(0, 0.02) + feature_influence * 0.01
                price = prices[-1] * (1 + price_change)
            
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price * (1 + np.random.normal(0, 0.005))
            close = price
            
            prices.append([open_price, high, low, close])
            volumes.append(np.random.lognormal(10, 1))
        
        return pd.DataFrame({
            'open': [p[0] for p in prices],
            'high': [p[1] for p in prices],
            'low': [p[2] for p in prices],
            'close': [p[3] for p in prices],
            'volume': volumes
        })


# Factory functions
def create_hybrid_cvlsa_tree_model(cvlsa_config: Optional[EnhancedCVLSAConfig] = None,
                                 tree_config: Optional[TreeCLVSAConfig] = None,
                                 tree_model_type: str = 'random_forest',
                                 fusion_method: str = 'weighted_average') -> HybridCVLSATreeModel:
    """Create hybrid CVLSA-tree model."""
    return HybridCVLSATreeModel(
        cvlsa_config=cvlsa_config,
        tree_config=tree_config,
        tree_model_type=tree_model_type,
        fusion_method=fusion_method
    )


def create_cvlsa_feature_extractor(cvlsa_config: Optional[EnhancedCVLSAConfig] = None) -> CVLSAFeatureExtractor:
    """Create CVLSA feature extractor."""
    return CVLSAFeatureExtractor(cvlsa_config=cvlsa_config)