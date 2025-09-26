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
import time
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

# Import existing components
from .cvlsa_architecture import (
    EnhancedCVLSAConfig, EnhancedCVLSATrainer, create_enhanced_cvlsa_model
)
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
                 tree_model_type: str = 'random_forest',
                 fusion_method: str = 'weighted_average',
                 cvlsa_weight: float = 0.6,
                 tree_weight: float = 0.4):
        """
        Initialize hybrid CVLSA-tree model.
        
        Args:
            cvlsa_config: CVLSA configuration
            tree_model_type: Type of tree model ('random_forest', 'extra_trees', 'xgboost', 'lightgbm', 'catboost')
            fusion_method: Method to fuse CVLSA and tree predictions ('weighted_average', 'stacking', 'attention')
            cvlsa_weight: Weight for CVLSA predictions in fusion
            tree_weight: Weight for tree predictions in fusion
        """
        self.cvlsa_config = cvlsa_config or EnhancedCVLSAConfig()
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
                # For now, just fit with enhanced features (regime-aware training can be added later)
                self.tree_model.fit(enhanced_features_scaled, y)
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
        if hasattr(self.tree_model, 'feature_importances_'):
            importance['tree_importance'] = self.tree_model.feature_importances_
        
        return importance
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_type': 'HybridCVLSATree',
            'cvlsa_config': self.cvlsa_config.__dict__,
            'tree_model_type': self.tree_model_type,
            'fusion_method': self.fusion_method,
            'is_fitted': self.is_fitted,
            'training_metadata': self.training_metadata
        }


class CVLSAFeatureExtractor:
    """
    Advanced feature extractor that uses CVLSA to enhance features for any downstream model.
    Integrates with caching system for efficient reuse across multiple models and datasets.
    """

    def __init__(self, cvlsa_config: Optional[EnhancedCVLSAConfig] = None):
        self.cvlsa_config = cvlsa_config or EnhancedCVLSAConfig()
        self.cvlsa_model = None
        self.feature_scaler = StandardScaler()
        self.is_fitted = False

        # Caching integration
        self.cache_manager = None
        self.feature_cache = {}

        # Configuration for automatic operation
        self.auto_enhance = True
        self.max_cache_size = 50
        self.memory_limit_mb = 200.0

        logger.info("🔧 CVLSA Feature Extractor initialized")

    def _init_cache_manager(self):
        """Initialize cache manager for feature extraction."""
        if self.cache_manager is None:
            from src.utils.ml_common.models.cvlsa_cache import CLVSACacheConfig, get_global_clvsa_cache

            cache_config = CLVSACacheConfig(
                max_cache_size=self.max_cache_size,
                max_memory_mb=self.memory_limit_mb,
                ttl_seconds=1800,  # 30 minutes
                enable_persistence=True,
                cache_dir="./cvlsa_feature_cache"
            )
            self.cache_manager = get_global_clvsa_cache(cache_config)

    def _generate_feature_config(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate configuration key for caching."""
        return {
            'input_dim': market_data.shape[1] + 1,  # +1 for target
            'output_dim': 4,
            'seq_length': len(market_data),
            'cross_view_attention': self.cvlsa_config.cross_view_attention,
            'use_multi_scale_attention': self.cvlsa_config.use_multi_scale_attention,
            'memory_efficient': self.cvlsa_config.memory_efficient,
            'use_m1_gpu': self.cvlsa_config.use_m1_gpu,
            'attention_dim': self.cvlsa_config.view_embedding_dim
        }

    def _prepare_enhanced_market_data(self, X: np.ndarray, market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Prepare enhanced market data with technical indicators."""
        if market_data is not None:
            return self._add_technical_indicators(market_data)

        # Create synthetic market data with enhanced features
        return self._create_enhanced_synthetic_market_data(X)

    def _add_technical_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to market data."""
        enhanced_data = market_data.copy()

        # Add moving averages
        close_prices = market_data['close'].values
        for window in [5, 10, 20, 50]:
            ma = pd.Series(close_prices).rolling(window).mean().fillna(close_prices[0])
            enhanced_data[f'sma_{window}'] = ma

        # Add RSI
        enhanced_data['rsi_14'] = self._calculate_rsi(close_prices, 14)

        # Add MACD
        macd_line, signal_line, histogram = self._calculate_macd(close_prices)
        enhanced_data['macd'] = macd_line
        enhanced_data['macd_signal'] = signal_line
        enhanced_data['macd_histogram'] = histogram

        # Add Bollinger Bands
        sma_20 = pd.Series(close_prices).rolling(20).mean().fillna(close_prices[0])
        std_20 = pd.Series(close_prices).rolling(20).std().fillna(0)
        enhanced_data['bb_upper'] = sma_20 + (std_20 * 2)
        enhanced_data['bb_middle'] = sma_20
        enhanced_data['bb_lower'] = sma_20 - (std_20 * 2)

        # Add volume indicators
        if 'volume' in market_data.columns:
            volumes = market_data['volume'].values
            enhanced_data['volume_ma_5'] = pd.Series(volumes).rolling(5).mean().fillna(volumes[0])
            enhanced_data['volume_ma_10'] = pd.Series(volumes).rolling(10).mean().fillna(volumes[0])

        return enhanced_data

    def _create_enhanced_synthetic_market_data(self, X: np.ndarray) -> pd.DataFrame:
        """Create enhanced synthetic market data with technical indicators."""
        n_samples = X.shape[0]
        base_price = 100.0
        prices = []
        volumes = []

        for i in range(n_samples):
            if i == 0:
                price = base_price
            else:
                # Use more features to influence price movement
                feature_influences = []
                for j in range(min(10, X.shape[1])):  # Use first 10 features
                    feature_influences.append(X[i, j])

                feature_influence = np.mean(feature_influences) if feature_influences else 0
                price_change = np.random.normal(0, 0.02) + feature_influence * 0.01
                price = prices[-1] * (1 + price_change)

            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price * (1 + np.random.normal(0, 0.005))
            close = price

            prices.append([open_price, high, low, close])
            volumes.append(np.random.lognormal(10, 1))

        market_data = pd.DataFrame({
            'open': [p[0] for p in prices],
            'high': [p[1] for p in prices],
            'low': [p[2] for p in prices],
            'close': [p[3] for p in prices],
            'volume': volumes
        })

        return self._add_technical_indicators(market_data)

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(period).mean().fillna(0).values
        avg_loss = pd.Series(loss).rolling(period).mean().fillna(0).values

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD indicator."""
        ema_fast = pd.Series(prices).ewm(span=fast).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow).mean().values

        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def fit(self, X: np.ndarray, y: np.ndarray,
            market_data: Optional[pd.DataFrame] = None) -> 'CVLSAFeatureExtractor':
        """Fit the CVLSA feature extractor with caching."""
        logger.info("🔧 Training enhanced CVLSA feature extractor...")

        # Initialize cache manager
        self._init_cache_manager()

        # Prepare enhanced market data
        enhanced_market_data = self._prepare_enhanced_market_data(X, market_data)

        # Generate feature configuration for caching
        feature_config = self._generate_feature_config(enhanced_market_data)

        # Try to retrieve from cache first
        cached_result = self.cache_manager.retrieve(enhanced_market_data, feature_config)

        if cached_result is not None:
            features, predictions, attention_weights = cached_result
            logger.info("🎯 Retrieved CVLSA features from cache")

            # Create CVLSA model for this configuration
            self.cvlsa_model = EnhancedCVLSATrainer(self.cvlsa_config)

            # Store cached data for transform method
            self.cached_features = features
            self.cached_predictions = predictions
            self.enhanced_market_data = enhanced_market_data

            self.is_fitted = True
            logger.info("✅ CVLSA feature extractor loaded from cache")
            return self

        # Not in cache, compute from scratch
        logger.info("🔧 Computing CVLSA features (cache miss)")

        # Initialize and train CVLSA model
        self.cvlsa_model = EnhancedCVLSATrainer(self.cvlsa_config)
        cvlsa_features = self.cvlsa_model.prepare_features(enhanced_market_data)

        # Create target tensor
        target = torch.FloatTensor(enhanced_market_data['close'].values)

        # Train CVLSA model
        cvlsa_results = self.cvlsa_model.train(cvlsa_features, cvlsa_features, target)

        # Get predictions and attention weights for caching
        with torch.no_grad():
            predictions = self.cvlsa_model.predict(cvlsa_features)

        attention_weights = self.cvlsa_model.get_attention_weights()

        # Store in cache
        cache_key = self.cache_manager.store(
            enhanced_market_data, feature_config, cvlsa_features, predictions, attention_weights
        )

        # Store for transform method
        self.cached_features = cvlsa_features
        self.cached_predictions = predictions
        self.enhanced_market_data = enhanced_market_data

        self.is_fitted = True
        logger.info(f"✅ CVLSA feature extractor trained and cached (key: {cache_key[:8]}...)")

        return self

    def transform(self, X: np.ndarray, market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Extract enhanced features using CVLSA with caching."""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transforming")

        # Prepare market data (same as in fit)
        enhanced_market_data = self._prepare_enhanced_market_data(X, market_data)

        # Check if market data has changed significantly
        if not enhanced_market_data.equals(self.enhanced_market_data):
            logger.warning("⚠️ Market data has changed, refitting may be required")
            # For now, we'll use the cached model but log the warning
            pass

        # Get CVLSA predictions (already cached)
        with torch.no_grad():
            cvlsa_features_np = self.cached_predictions.cpu().numpy()

        # Combine original features with CVLSA features
        enhanced_features = np.hstack([X, cvlsa_features_np])

        # Scale features
        enhanced_features_scaled = self.feature_scaler.fit_transform(enhanced_features)

        return enhanced_features_scaled

    def fit_transform(self, X: np.ndarray, y: np.ndarray,
                     market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Fit and transform in one step with caching."""
        return self.fit(X, y, market_data).transform(X, market_data)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from CVLSA attention weights."""
        if self.cvlsa_model is None:
            return {}

        attention_weights = self.cvlsa_model.get_attention_weights()

        # Calculate feature importance based on attention weights
        importance = {}

        if 'cross_view' in attention_weights:
            # Average attention weights across all heads and sequences
            avg_attention = np.mean(attention_weights['cross_view'], axis=(0, 1))
            importance['cvlsa_attention'] = avg_attention

        return importance

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache_manager is None:
            return {}

        return self.cache_manager.get_stats()

    def clear_cache(self):
        """Clear the feature cache."""
        if self.cache_manager is not None:
            self.cache_manager.clear()
            logger.info("🧹 CVLSA feature cache cleared")

    def set_cache_config(self, max_cache_size: int = 50, memory_limit_mb: float = 200.0):
        """Update cache configuration."""
        self.max_cache_size = max_cache_size
        self.memory_limit_mb = memory_limit_mb

        if self.cache_manager is not None:
            logger.info(f"🔄 Updated cache configuration: size={max_cache_size}, memory={memory_limit_mb}MB")


class AutomaticCVLSAFeaturePipeline:
    """
    Automatic feature extraction pipeline that integrates CVLSA with any ML training process.
    This pipeline automatically enhances features using CVLSA and caches results for reuse.
    """

    def __init__(self, cvlsa_config: Optional[EnhancedCVLSAConfig] = None):
        self.feature_extractor = CVLSAFeatureExtractor(cvlsa_config)
        self.enabled = True
        self.auto_enhance = True
        self.verbose = True

        # Pipeline configuration
        self.add_technical_indicators = True
        self.enhance_synthetic_data = True
        self.cache_features = True

        logger.info("🔧 Automatic CVLSA Feature Pipeline initialized")

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None,
                     market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Fit the feature extractor and transform features automatically."""
        if not self.enabled:
            logger.info("⚠️ Automatic CVLSA feature pipeline is disabled")
            return X

        logger.info("🚀 Starting automatic CVLSA feature enhancement...")

        if y is None:
            logger.warning("⚠️ No target provided, using unsupervised feature extraction")
            y = np.zeros(len(X))  # Dummy target for unsupervised learning

        # Fit and transform features
        enhanced_features = self.feature_extractor.fit_transform(X, y, market_data)

        if self.verbose:
            original_shape = X.shape
            enhanced_shape = enhanced_features.shape
            logger.info(f"✅ Feature enhancement completed: {original_shape[1]} → {enhanced_shape[1]} features")
            logger.info(f"📊 Enhanced features include: Original + CVLSA predictions + Technical indicators")

        return enhanced_features

    def transform(self, X: np.ndarray, market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Transform features using fitted extractor."""
        if not self.enabled:
            return X

        return self.feature_extractor.transform(X, market_data)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from the CVLSA model."""
        return self.feature_extractor.get_feature_importance()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.feature_extractor.get_cache_stats()

    def clear_cache(self):
        """Clear the feature cache."""
        self.feature_extractor.clear_cache()

    def set_cache_config(self, max_cache_size: int = 50, memory_limit_mb: float = 200.0):
        """Update cache configuration."""
        self.feature_extractor.set_cache_config(max_cache_size, memory_limit_mb)

    def enable_auto_enhancement(self, enable: bool = True):
        """Enable or disable automatic feature enhancement."""
        self.enabled = enable
        self.auto_enhance = enable
        logger.info(f"🔄 Auto-enhancement {'enabled' if enable else 'disabled'}")


class CVLSAFeatureEnhancer:
    """
    High-level feature enhancer that can be used as a preprocessing step in any ML pipeline.
    Automatically detects data types and applies appropriate CVLSA enhancements.
    """

    def __init__(self, auto_detect: bool = True, enhancement_level: str = 'comprehensive'):
        self.auto_detect = auto_detect
        self.enhancement_level = enhancement_level  # 'basic', 'comprehensive', 'advanced'

        # Initialize appropriate pipeline based on enhancement level
        cvlsa_config = self._create_cvlsa_config_for_level(enhancement_level)
        self.pipeline = AutomaticCVLSAFeaturePipeline(cvlsa_config)

        # Auto-detection settings
        self.market_data_columns = ['open', 'high', 'low', 'close', 'volume']
        self.technical_indicators = [
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'bb_upper', 'bb_middle', 'bb_lower'
        ]

        logger.info(f"🔧 CVLSA Feature Enhancer initialized (level: {enhancement_level})")

    def _create_cvlsa_config_for_level(self, level: str) -> EnhancedCVLSAConfig:
        """Create CVLSA configuration based on enhancement level."""
        base_config = EnhancedCVLSAConfig()

        if level == 'basic':
            base_config.view_embedding_dim = 32
            base_config.cross_attention_heads = 4
            base_config.temporal_attention_heads = 4
            base_config.memory_efficient = True
        elif level == 'comprehensive':
            base_config.view_embedding_dim = 64
            base_config.cross_attention_heads = 8
            base_config.temporal_attention_heads = 8
            base_config.memory_efficient = True
        elif level == 'advanced':
            base_config.view_embedding_dim = 128
            base_config.cross_attention_heads = 16
            base_config.temporal_attention_heads = 16
            base_config.memory_efficient = False  # Use more memory for better performance
            base_config.gradient_checkpointing = False

        return base_config

    def _detect_market_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Auto-detect market data from input."""
        if isinstance(data, pd.DataFrame):
            # Check if DataFrame contains market data columns
            market_cols = [col for col in self.market_data_columns if col in data.columns]
            if len(market_cols) >= 3:  # At least open, high, low, close
                return data[market_cols].copy()

        return None

    def _is_financial_data(self, X: np.ndarray) -> bool:
        """Determine if the input data appears to be financial data."""
        # Simple heuristics - financial data often has specific statistical properties
        if X.shape[1] < 5:
            return False

        # Check for typical financial data patterns (returns, volatility, etc.)
        returns_std = np.std(np.diff(X, axis=0), axis=0).mean()
        if returns_std > 0.1:  # High volatility suggests financial data
            return True

        return False

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None,
                     market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Automatically detect and enhance features."""
        if not self.pipeline.enabled:
            logger.info("⚠️ Feature enhancement is disabled")
            return X

        logger.info("🔍 Auto-detecting data type and applying appropriate enhancements...")

        # Auto-detect market data if not provided
        if market_data is None and self.auto_detect:
            market_data = self._detect_market_data(X)
            if market_data is not None:
                logger.info(f"📊 Auto-detected market data with {market_data.shape[1]} columns")
            else:
                logger.info("📊 No market data detected, will create synthetic data")

        # Determine if this is financial data for appropriate enhancement
        is_financial = self._is_financial_data(X)
        if is_financial:
            logger.info("📈 Financial data detected, applying comprehensive enhancement")
        else:
            logger.info("📊 General data detected, applying standard enhancement")

        # Apply enhancement
        enhanced_features = self.pipeline.fit_transform(X, y, market_data)

        # Add metadata about enhancement
        self.enhancement_metadata = {
            'original_features': X.shape[1],
            'enhanced_features': enhanced_features.shape[1],
            'enhancement_level': self.enhancement_level,
            'market_data_detected': market_data is not None,
            'is_financial_data': is_financial
        }

        return enhanced_features

    def transform(self, X: np.ndarray, market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Transform new data using fitted enhancer."""
        return self.pipeline.transform(X, market_data)

    def get_enhancement_info(self) -> Dict[str, Any]:
        """Get information about the enhancement applied."""
        if hasattr(self, 'enhancement_metadata'):
            return self.enhancement_metadata
        else:
            return {'status': 'not_fitted'}

    def enable_enhancement(self, enable: bool = True):
        """Enable or disable feature enhancement."""
        self.pipeline.enable_auto_enhancement(enable)


# Factory functions
def create_hybrid_cvlsa_tree_model(cvlsa_config: Optional[EnhancedCVLSAConfig] = None,
                                 tree_model_type: str = 'random_forest',
                                 fusion_method: str = 'weighted_average') -> HybridCVLSATreeModel:
    """Create hybrid CVLSA-tree model."""
    return HybridCVLSATreeModel(
        cvlsa_config=cvlsa_config,
        tree_model_type=tree_model_type,
        fusion_method=fusion_method
    )


def create_cvlsa_feature_extractor(cvlsa_config: Optional[EnhancedCVLSAConfig] = None) -> CVLSAFeatureExtractor:
    """Create CVLSA feature extractor."""
    return CVLSAFeatureExtractor(cvlsa_config=cvlsa_config)


def create_automatic_feature_pipeline(cvlsa_config: Optional[EnhancedCVLSAConfig] = None) -> AutomaticCVLSAFeaturePipeline:
    """Create automatic CVLSA feature pipeline."""
    return AutomaticCVLSAFeaturePipeline(cvlsa_config)


def create_feature_enhancer(auto_detect: bool = True, enhancement_level: str = 'comprehensive') -> CVLSAFeatureEnhancer:
    """Create high-level feature enhancer with auto-detection."""
    return CVLSAFeatureEnhancer(auto_detect, enhancement_level)