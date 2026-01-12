"""
Train Specialists with GMM Enhancement Step

This step trains all 11 specialist models independently and processes their
outputs through GMM enhanced features for downstream use.

Specialists Trained:
- Momentum Persistence: Captures structural inertia and trend sustainability
- SMC Regime: Smart Money Concepts focused on order blocks and liquidity sweeps  
- Volatility Burst: Detects compression regimes and imminent expansion
- Volume Force: Binary breakout classifier focused on order-flow impulse
- Macro Regime: High-horizon trend and regime shift detection
- Meso Regime: Intermediate-horizon cyclical and trend patterns
- Liquidity Regime: Monitors market depth and capacity states
- Path Regime: Analyzes the "roughness" and risk of the price path
- Risk Regime: Focuses on tail-risk (VaR/CVaR) and volatility escalation
- Microstructure: Analyzes spread volatility, price efficiency, and imbalance
- Spectral Energy: Captures frequency-domain energy and dominant cycles
"""

import logging
import os
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import gc
import asyncio
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import psutil
import numba
from numba import jit, prange
import warnings
import pickle
import hashlib
from dataclasses import dataclass
warnings.filterwarnings('ignore')

# Importations pour TreeSHAP
try:
    import shap
    import xgboost as xgb
    TREESHAP_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"⚠️ TreeSHAP not available: {e}")
    TREESHAP_AVAILABLE = False

from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.training.steps.market_analysis.afml_specialist_mixin import AFMLSpecialistMixin
from src.training.steps.market_analysis.gmm_report_generator import GMMReportGenerator
from src.utils.data.entropy_bars import build_entropy_bars_15min

# Import all enhanced specialists
SPECIALIST_IMPORTS = {
    "enhanced_ml_momentum_persistence_step": ("src.training.steps.market_analysis.ml_momentum_persistence_step_enhanced", "EnhancedMLMomentumPersistenceStep"),
    "enhanced_ml_smc_regime_step": ("src.training.steps.market_analysis.ml_smc_regime_step_enhanced", "EnhancedMLSMCRegimeStep"),
    "enhanced_ml_volatility_burst_step": ("src.training.steps.market_analysis.ml_volatility_burst_step_enhanced", "EnhancedMLVolatilityBurstStep"),
    "enhanced_ml_volume_force_step": ("src.training.steps.market_analysis.ml_volume_force_step_enhanced", "EnhancedMLVolumeForceStep"),
    # "enhanced_xgb_macro_regime_step": ("src.training.steps.market_analysis.xgb_macro_regime_step_enhanced", "EnhancedXGBMacroRegimeStep"),
    # "enhanced_xgb_meso_regime_step": ("src.training.steps.market_analysis.xgb_meso_regime_step_enhanced", "EnhancedXGBMesoRegimeStep"),
    # "enhanced_ml_liquidity_regime_step": ("src.training.steps.market_analysis.ml_liquidity_regime_step_enhanced", "EnhancedMLLiquidityRegimeStep"),
    # "enhanced_ml_path_regime_step": ("src.training.steps.market_analysis.ml_path_regime_step_enhanced", "EnhancedMLPathRegimeStep"),
    # "enhanced_ml_risk_regime_step": ("src.training.steps.market_analysis.ml_risk_regime_step_enhanced", "EnhancedMLRiskRegimeStep"),
    # "enhanced_ml_microstructure_step": ("src.training.steps.market_analysis.ml_microstructure_step_enhanced", "EnhancedMLMicrostructureStep"),
    # "enhanced_ml_spectral_step": ("src.training.steps.market_analysis.ml_spectral_step_enhanced", "EnhancedMLSpectralStep"),
    # "enhanced_ml_candlestick_step": ("src.training.steps.market_analysis.ml_candlestick_step_enhanced", "EnhancedMLCandlestickStep"),
    # "enhanced_ml_reversion_regime_step": ("src.training.steps.market_analysis.ml_reversion_regime_step_enhanced", "EnhancedMLReversionRegimeStep"),
}

# Import GMM enhanced features
try:
    from src.training.steps.market_analysis.gmm_enhanced_features import EnhancedGMMFeatures
    GMM_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"⚠️ GMM Enhanced Features not available: {e}")
    GMM_AVAILABLE = False


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    max_memory_usage: float = 0.8  # 80% of available RAM
    memory_pool_size: int = 100  # Maximum arrays in memory pool
    gc_frequency: int = 5  # GC every N batches
    enable_memory_mapping: bool = True
    sparse_threshold: float = 0.7  # Sparsity threshold for sparse arrays

@dataclass
class GMMConfig:
    """GMM model configuration."""
    model_cache_dir: str = "artifacts/gmm_models"
    enable_persistence: bool = True
    enable_incremental: bool = True
    adaptation_threshold: float = 0.1  # Performance threshold for retraining
    min_samples_for_update: int = 1000
    max_models_cached: int = 10

@dataclass
class BatchConfig:
    """Dynamic batch configuration."""
    base_batch_size: int = 3
    min_batch_size: int = 1
    max_batch_size: int = 8
    memory_factor: float = 0.3  # GB per batch
    cpu_factor: float = 2.0  # Cores per batch
    specialist_complexity: Dict[str, float] = None


class TrainSpecialistsWithGMMStep(BaseStep):
    """
    Step to train all specialist models and enhance their outputs with GMM.
    
    This step:
    1. Trains each specialist independently
    2. Collects specialist outputs/predictions
    3. Processes outputs through GMM enhanced features
    4. Saves enhanced features for downstream use
    """

    def __init__(self, step_name: str = "train_specialists_with_gmm", **kwargs):
        super().__init__(step_name, **kwargs)
        self.artifacts_dir = Path("artifacts/specialists_with_gmm")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Metrics collection for comprehensive specialist evaluation
        self._specialist_metrics = {}  # Store detailed metrics for each specialist
        self._specialist_outputs = {}  # Store outputs for each specialist

        # #region agent log - Hypothesis A: Memory management initialization
        import json
        import os
        with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "log_memory_init",
                "timestamp": int(__import__('time').time() * 1000),
                "location": "train_specialists_with_gmm_step.py:__init__",
                "message": "Initializing memory management configurations",
                "data": {"step_name": step_name, "kwargs_keys": list(kwargs.keys())},
                "sessionId": "debug-session",
                "runId": "initial",
                "hypothesisId": "A"
            }) + '\n')
        # #endregion

        # Initialize configurations
        self.memory_config = MemoryConfig()
        self.gmm_config = GMMConfig()
        self.batch_config = BatchConfig()

        # Initialize memory management
        self.memory_pool = {}
        self.batch_count = 0
        self.memory_usage_history = []

        # Initialize GMM model cache
        self.gmm_model_cache = {}
        self.gmm_model_metadata = {}

        # Ensure GMM model cache directory exists
        Path(self.gmm_config.model_cache_dir).mkdir(parents=True, exist_ok=True)

        # #region agent log - Hypothesis A: Memory management initialized
        with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "log_memory_init_complete",
                "timestamp": int(__import__('time').time() * 1000),
                "location": "train_specialists_with_gmm_step.py:__init__",
                "message": "Memory management initialization completed",
                "data": {
                    "memory_config": {"max_memory_usage": self.memory_config.max_memory_usage, "memory_pool_size": self.memory_config.memory_pool_size},
                    "gmm_config": {"enable_persistence": self.gmm_config.enable_persistence, "max_models_cached": self.gmm_config.max_models_cached},
                    "batch_config": {"base_batch_size": self.batch_config.base_batch_size, "max_batch_size": self.batch_config.max_batch_size}
                },
                "sessionId": "debug-session",
                "runId": "initial",
                "hypothesisId": "A"
            }) + '\n')
        # #endregion

    def _get_optimal_batch_size(self, base_batch_size: int = None) -> int:
        """Calculate optimal batch size based on system resources and specialist complexity."""
        if base_batch_size is None:
            base_batch_size = self.batch_config.base_batch_size
            
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
            cpu_cores = psutil.cpu_count()
            
            # Calculate memory-based batch size
            memory_batch_size = max(
                self.batch_config.min_batch_size,
                min(
                    self.batch_config.max_batch_size,
                    int(available_memory_gb / self.batch_config.memory_factor)
                )
            )
            
            # Calculate CPU-based batch size
            cpu_batch_size = max(
                self.batch_config.min_batch_size,
                min(
                    self.batch_config.max_batch_size,
                    int(cpu_cores / self.batch_config.cpu_factor)
                )
            )
            
            # Use the more conservative estimate
            optimal_batch = min(memory_batch_size, cpu_batch_size)
            
            # Apply memory pressure adjustment
            if len(self.memory_usage_history) > 5:
                recent_usage = np.mean(self.memory_usage_history[-5:])
                if recent_usage > self.memory_config.max_memory_usage:
                    optimal_batch = max(self.batch_config.min_batch_size, optimal_batch - 1)
                    tprint_warning(f"⚠️ High memory usage detected, reducing batch size to {optimal_batch}")
            
            self.memory_usage_history.append(memory.percent / 100.0)
            
            tprint_info(f"💻 System: {available_memory_gb:.1f}GB RAM, {cpu_cores} cores → batch size: {optimal_batch}")
            return optimal_batch
            
        except Exception as e:
            tprint_warning(f"⚠️ Could not detect system resources, using default batch size: {e}")
            return base_batch_size

    def _optimize_memory_usage(self) -> None:
        """Monitor and optimize memory usage during training."""
        try:
            memory = psutil.virtual_memory()
            current_usage = memory.percent / 100.0
            
            if current_usage > self.memory_config.max_memory_usage:
                tprint_warning(f"⚠️ Memory usage at {current_usage:.1%}, triggering cleanup")
                
                # Clear memory pool
                self.memory_pool.clear()
                
                # Clear GMM model cache if needed
                if len(self.gmm_model_cache) > self.gmm_config.max_models_cached:
                    # Remove oldest models
                    oldest_models = list(self.gmm_model_cache.keys())[:-self.gmm_config.max_models_cached]
                    for model_key in oldest_models:
                        del self.gmm_model_cache[model_key]
                        if model_key in self.gmm_model_metadata:
                            del self.gmm_model_metadata[model_key]
                    tprint_info(f"🧹 Cleared {len(oldest_models)} old GMM models from cache")
                
                # Force garbage collection
                gc.collect()
                
                # Check memory after cleanup
                memory_after = psutil.virtual_memory()
                usage_after = memory_after.percent / 100.0
                tprint_info(f"✅ Memory usage reduced to {usage_after:.1%}")
                
        except Exception as e:
            tprint_warning(f"⚠️ Memory optimization failed: {e}")
    
    def _get_memory_pool_array(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Get array from memory pool or create new one."""
        try:
            array_key = f"{shape}_{dtype}"
            
            if array_key in self.memory_pool:
                array = self.memory_pool[array_key]
                if array.shape == shape and array.dtype == dtype:
                    # Clear existing data
                    array.fill(0)
                    return array
            
            # Create new array if not in pool or incompatible
            array = np.zeros(shape, dtype=dtype)
            
            # Add to pool if space available
            if len(self.memory_pool) < self.memory_config.memory_pool_size:
                self.memory_pool[array_key] = array.copy()
            
            return array
            
        except Exception as e:
            tprint_warning(f"⚠️ Memory pool access failed: {e}")
            return np.zeros(shape, dtype=dtype)
    
    def _generate_data_hash(self, data: pd.DataFrame) -> str:
        """Generate hash for data to identify cached models."""
        try:
            # Use shape, columns, and first/last few rows for hash
            hash_data = {
                'shape': data.shape,
                'columns': list(data.columns),
                'head': data.head().values.tobytes(),
                'tail': data.tail().values.tobytes(),
                'dtypes': data.dtypes.to_dict()
            }
            hash_string = str(sorted(hash_data.items())).encode()
            return hashlib.md5(hash_string).hexdigest()
        except Exception as e:
            tprint_warning(f"⚠️ Hash generation failed: {e}")
            return f"fallback_{len(data)}_{len(data.columns)}"

    def _perform_treeshap_analysis(self, features: pd.DataFrame, target: pd.Series, max_features: int = 100) -> List[str]:
        """Perform TreeSHAP analysis to identify most important features."""
        if not TREESHAP_AVAILABLE:
            tprint_warning("⚠️ TreeSHAP not available, falling back to mutual information")
            return None
        
        try:
            tprint_info("🌲 Performing TreeSHAP analysis for feature selection...")
            start_time = time.time()
            
            # Prepare data for XGBoost
            X = features.fillna(0).values
            y = target.values
            
            # Train a simple XGBoost model for SHAP analysis
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'tree_method': 'hist',
                'device': 'cpu'
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X, y)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Calculate mean absolute SHAP values for each feature
            shap_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Create feature-importance mapping
            feature_importance = list(zip(features.columns, shap_importance))
            
            # Sort by importance (descending)
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Select top features
            selected_features = [col for col, _ in feature_importance[:max_features]]
            
            elapsed_time = time.time() - start_time
            tprint_success(f"✅ TreeSHAP analysis completed in {elapsed_time:.2f}s")
            tprint_info(f"📊 Top 5 features by SHAP importance: {[col for col, _ in feature_importance[:5]]}")
            
            return selected_features
            
        except Exception as e:
            tprint_error(f"❌ TreeSHAP analysis failed: {e}")
            return None
    
    def _save_gmm_model(self, model_key: str, model: Any, metadata: Dict[str, Any]) -> None:
        """Save GMM model to disk with metadata."""
        try:
            if not self.gmm_config.enable_persistence:
                return
                
            model_path = Path(self.gmm_config.model_cache_dir) / f"{model_key}.pkl"
            metadata_path = Path(self.gmm_config.model_cache_dir) / f"{model_key}_metadata.json"
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata
            metadata['saved_at'] = datetime.now().isoformat()
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            tprint_info(f"💾 Saved GMM model: {model_key}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save GMM model {model_key}: {e}")
    
    def _load_gmm_model(self, model_key: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """Load GMM model from disk with metadata."""
        try:
            if not self.gmm_config.enable_persistence:
                return None, None
                
            model_path = Path(self.gmm_config.model_cache_dir) / f"{model_key}.pkl"
            metadata_path = Path(self.gmm_config.model_cache_dir) / f"{model_key}_metadata.json"
            
            if not model_path.exists() or not metadata_path.exists():
                return None, None
            
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            tprint_info(f"📥 Loaded GMM model: {model_key}")
            return model, metadata
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load GMM model {model_key}: {e}")
            return None, None
    
    def _should_update_model(self, metadata: Dict[str, Any], new_data_size: int) -> bool:
        """Determine if GMM model should be updated based on new data."""
        try:
            if not self.gmm_config.enable_incremental:
                return False
                
            # Check if enough new data
            if new_data_size < self.gmm_config.min_samples_for_update:
                return False
            
            # Check model age
            if 'saved_at' in metadata:
                saved_time = datetime.fromisoformat(metadata['saved_at'])
                age_hours = (datetime.now() - saved_time).total_seconds() / 3600
                
                # Update if model is older than 24 hours and we have new data
                if age_hours > 24:
                    return True
            
            # Check performance degradation (if available)
            if 'performance_score' in metadata:
                performance_score = metadata['performance_score']
                if performance_score < self.gmm_config.adaptation_threshold:
                    return True
            
            return False
            
        except Exception as e:
            tprint_warning(f"⚠️ Model update check failed: {e}")
            return True  # Default to updating if check fails
    
    def _calculate_adaptive_parameters(self, market_data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate adaptive GMM parameters based on data characteristics."""
        try:
            # Data characteristics
            n_samples = len(market_data)
            n_features = len(features.columns)
            data_volatility = market_data['close'].pct_change().std()
            
            # Adaptive n_components based on data size and complexity
            if n_samples < 1000:
                n_components = 4
            elif n_samples < 10000:
                n_components = 6
            elif n_samples < 50000:
                n_components = 8
            else:
                n_components = min(12, max(4, int(np.sqrt(n_features) * 2)))
            
            # Adaptive subsample size
            subsample_size = min(20000, max(5000, n_samples // 10))
            
            # Adaptive wavelet based on volatility
            if data_volatility < 0.01:
                wavelet = 'db4'
            elif data_volatility < 0.02:
                wavelet = 'db6'
            else:
                wavelet = 'db8'
            
            # Adaptive permutation count based on data size
            n_permutations = min(100, max(20, n_samples // 1000))
            
            # Adaptive fracdiff parameters
            if data_volatility < 0.015:
                max_d = 0.8
                min_d = 0.0
            else:
                max_d = 1.2
                min_d = 0.2
            
            adaptive_config = {
                'n_components': n_components,
                'n_neighbors': 3,
                'subsample_size': subsample_size,
                'wavelet': wavelet,
                'n_permutations': n_permutations,
                'fracdiff_config': {
                    'max_d': max_d,
                    'min_d': min_d,
                    'adf_threshold': 0.01,
                    'method': 'binary_search',
                    'tolerance': 0.01
                },
                'har_windows': [1, 5, 22],
                'shock_window': min(50, max(10, n_samples // 100))
            }
            
            tprint_info(f"🎯 Adaptive parameters: n_components={n_components}, subsample={subsample_size}, wavelet={wavelet}")
            return adaptive_config
            
        except Exception as e:
            tprint_warning(f"⚠️ Adaptive parameter calculation failed: {e}")
            # Return default parameters
            return {
                'n_components': 8,
                'n_neighbors': 3,
                'subsample_size': min(20000, len(market_data)),
                'wavelet': 'db4',
                'n_permutations': 50,
                'fracdiff_config': {
                    'max_d': 1.0,
                    'min_d': 0.0,
                    'adf_threshold': 0.01,
                    'method': 'binary_search',
                    'tolerance': 0.01
                },
                'har_windows': [1, 5, 22],
                'shock_window': 20
            }

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute specialist training with GMM enhancement.

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary containing training results and enhanced features
        """
        start_time = time.time()
        tprint_info("🚀 Starting Specialist Training with GMM Enhancement Pipeline")
        
        # Extract configuration
        symbol = config.get("symbol", "ETHUSDT")
        exchange = config.get("exchange", "binance") 
        timeframe = config.get("timeframe", "15m")
        direction = config.get("direction", "long")
        force_retrain = config.get("force_retrain", False)
        
        tprint_info(f"📊 Configuration: {symbol} {exchange} {timeframe} {direction}")
        
        # Step 1: Load market data with memory optimization
        tprint_info("📥 Loading market data...")
        market_data, source = self.load_market_data_or_fail(config, skip_artifacts=True)
        if market_data is None or market_data.empty:
            error_msg = "❌ Failed to load market data"
            tprint_error(error_msg)
            return {"success": False, "error": error_msg}

        tprint_success(f"✅ Loaded {len(market_data)} bars from {source}")

        # --- Entropy Bar Conversion ---
        # User Requirement: If time bars, switch to entropy bars (target 1/15min)
        # Check if index is time-based (proxy for time bars)
        if isinstance(market_data.index, pd.DatetimeIndex):
            tprint_info("✨ User requested Entropy Bars. Converting Time Bars (15m) -> Entropy Bars...")
            try:
                entropy_bars, threshold = build_entropy_bars_15min(market_data)
                
                if not entropy_bars.empty:
                    old_len = len(market_data)
                    new_len = len(entropy_bars)
                    tprint_success(f"✅ Converted to Entropy Bars: {old_len} -> {new_len} bars (Threshold={threshold:.4f})")
                    market_data = entropy_bars
                else:
                    tprint_warning("⚠️ Entropy bar conversion yielded empty result. Keeping original bars.")
            except Exception as e:
                tprint_error(f"❌ Entropy Bar conversion failed: {e}. Keeping original bars.")

        # Memory optimization: reduce data size for training
        # Use only recent data to reduce memory usage
        max_training_samples = 25000  # Further reduced to 25k for M1 stability
        active_market_data = market_data
        
        if len(active_market_data) > max_training_samples:
            tprint_info(f"🧠 Reducing data size from {len(active_market_data)} to {max_training_samples} samples for memory efficiency")
            active_market_data = active_market_data.tail(max_training_samples)
            tprint_success(f"✅ Reduced data to {len(active_market_data)} samples")

        # Step 1.5: Generate enhanced features for all specialists
        tprint_info("🔧 Generating Enhanced Features for All Specialists...")
        enhanced_market_data = self._generate_enhanced_features(active_market_data)
        tprint_success(f"✅ Enhanced features generated: {enhanced_market_data.shape}")
        
        # Memory optimization: Clear references and collect garbage after heavy feature generation
        import gc
        gc.collect()



        # Step 2: Train all specialists and collect per-specialist features
        tprint_info("🎯 Training 11 Specialist Models and collecting features...")
        
        # Use very small batch size to prevent memory issues
        optimal_batch_size = 1  # Process one specialist at a time

        # #region agent log - Hypothesis D: Batch size calculated
        with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "log_batch_size_result",
                "timestamp": int(__import__('time').time() * 1000),
                "location": "train_specialists_with_gmm_step.py:execute",
                "message": "Batch size calculation completed",
                "data": {"optimal_batch_size": optimal_batch_size, "batch_config": {"base_batch_size": self.batch_config.base_batch_size, "max_batch_size": self.batch_config.max_batch_size}},
                "sessionId": "debug-session",
                "runId": "initial",
                "hypothesisId": "D"
            }) + '\n')
        # #endregion

        # Initialize memory monitoring
        self.batch_count = 0
        
        # Step 2: Extract raw features from market data (without training specialists)
        tprint_info("🔍 Extracting raw features from market data...")
        raw_features = await self._extract_raw_features_from_market_data(enhanced_market_data, config)
        tprint_success(f"✅ Extracted raw features: {raw_features.shape}")
        
        if raw_features is None or raw_features.empty:
            error_msg = "❌ No raw features extracted"
            tprint_error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Step 3: Apply GMM enhancement to raw features
        gmm_enhanced_features = None
        if GMM_AVAILABLE:
            tprint_info("🧠 Applying GMM enhancement to raw features (before specialist training)...")
            gmm_enhanced_features = await self._apply_gmm_enhancement_to_features(
                raw_features, active_market_data, config
            )
            
            if gmm_enhanced_features is None or gmm_enhanced_features.empty:
                tprint_warning("⚠️ GMM enhancement failed, using raw features")
                gmm_enhanced_features = raw_features
            else:
                tprint_success(f"✅ GMM enhanced raw features: {gmm_enhanced_features.shape}")
        else:
            gmm_enhanced_features = raw_features
            tprint_info("⚠️ GMM not available, using raw features")
        
        # Step 4: Feature selection on GMM-enhanced features
        tprint_info("🎯 Applying feature selection to GMM-enhanced features...")
        selected_features = await self._apply_feature_selection(gmm_enhanced_features, config)
        
        if selected_features is None or selected_features.empty:
            error_msg = "❌ Feature selection failed"
            tprint_error(error_msg)
            return {"success": False, "error": error_msg}
        
        tprint_success(f"✅ Selected {len(selected_features.columns)} features from {len(gmm_enhanced_features.columns)} GMM-enhanced features")
        
        # Step 5: Train Ensemble models (ExtraTrees, CatBoost, LGBM, XGB) on selected features
        model_results = await self._train_models_on_selected_features(selected_features, active_market_data, config)
        
        # Step 6: Train individual specialists using GMM-enhanced features
        tprint_info("🔄 Training individual specialists with GMM-enhanced features...")
        specialist_outputs = await self._train_all_specialists_memory_efficient(
            enhanced_market_data, config, force_retrain, batch_size=optimal_batch_size
        )
        
        # Log post-selection characteristics
        selection_ratio = len(selected_features.columns) / len(gmm_enhanced_features.columns)
        tprint_success(f"✅ Selected GMM features shape: {selected_features.shape}")
        tprint_info(f"📈 Feature selection: {len(gmm_enhanced_features.columns)} → {len(selected_features.columns)} ({selection_ratio:.1%} kept)")
        
        post_selection_types = self._analyze_feature_types(selected_features)
        tprint_info(f"🏷️ Post-selection feature types: {post_selection_types}")
        
        # Show top selected features by type
        if 'gmm' in post_selection_types and post_selection_types['gmm'] > 0:
            gmm_selected = [col for col in selected_features.columns if any(term in col.lower() for term in ['gmm', 'cluster', 'regime', 'state', 'probability', 'entropy', 'familiarity'])]
            tprint_info(f"🧠 GMM features selected: {len(gmm_selected)}/{post_selection_types['gmm']}")
            tprint_info(f"   Sample GMM features: {gmm_selected[:3]}..." if len(gmm_selected) > 3 else f"   Sample GMM features: {gmm_selected}")
        
        # Step 7: Save results
        await self._save_results(
            selected_features, specialist_outputs, config,
            ensemble_results=model_results
        )
        
        # Create results dictionary for return
        results = {
            "success": True,
            "symbol": config.get("symbol", "ETHUSDT"),
            "timeframe": config.get("timeframe", "15m"),
            "n_specialists": len(specialist_outputs),
            "gmm_enhancement_applied": GMM_AVAILABLE,
            "feature_selection_applied": True,
            "selected_features_shape": selected_features.shape,
            "model_results": model_results
        }
        
        tprint_success("✅ GMM Specialist Training Pipeline Completed Successfully")
        return results

    async def _apply_gmm_to_each_specialist(
        self,
        specialist_features: pd.DataFrame,
        market_data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply GMM enhancement to each specialist's features individually.
        
        This method processes each specialist's feature set separately through GMM,
        then combines the enhanced features for final processing.
        """
        try:
            tprint_info(f"🧠 Applying GMM enhancement to {len(specialist_features.columns)} total features...")
            
            # Split the combined features back into individual specialist feature sets
            specialist_feature_sets = {}
            
            # Extract specialist names from column prefixes
            specialist_names = set()
            for col in specialist_features.columns:
                # Find the specialist name (everything before the first underscore)
                if '_' in col:
                    specialist_name = col.split('_')[0]
                    specialist_names.add(specialist_name)
            
            tprint_info(f"📊 Found {len(specialist_names)} specialists: {list(specialist_names)}")
            
            # Extract features for each specialist
            for specialist_name in specialist_names:
                # Get columns that belong to this specialist
                specialist_cols = [col for col in specialist_features.columns if col.startswith(f"{specialist_name}_")]
                if specialist_cols:
                    # Remove prefix and create specialist-specific DataFrame
                    specialist_df = specialist_features[specialist_cols].copy()
                    specialist_df.columns = [col.replace(f"{specialist_name}_", "") for col in specialist_cols]
                    specialist_feature_sets[specialist_name] = specialist_df
                    tprint_info(f"✅ {specialist_name}: {len(specialist_cols)} features extracted")
                    
                    # Log feature characteristics
                    tprint_info(f"   📈 Feature types: {self._analyze_feature_types(specialist_df)}")
                    tprint_info(f"   🔢 Feature stats: mean={specialist_df.mean().mean():.4f}, std={specialist_df.std().mean():.4f}")
                    tprint_info(f"   📊 Sample features: {list(specialist_df.columns[:5])}")
            
            # Apply GMM enhancement to each specialist individually
            gmm_enhanced_specialists = {}
            enhancement_summary = []
            
            for specialist_name, features in specialist_feature_sets.items():
                tprint_info(f"🧠 Applying GMM to {specialist_name} ({features.shape})...")
                
                try:
                    # Log pre-GMM characteristics
                    tprint_info(f"   📊 Pre-GMM: {len(features.columns)} features, {len(features)} samples")
                    tprint_info(f"   🎯 Feature quality: missing={features.isnull().sum().sum()}, duplicates={features.columns.duplicated().sum()}")
                    
                    # Apply GMM enhancement to this specialist's features
                    enhanced_features = await self._apply_gmm_enhancement(
                        features, market_data, config
                    )
                    
                    if enhanced_features is not None and not enhanced_features.empty:
                        gmm_enhanced_specialists[specialist_name] = enhanced_features
                        
                        # Log post-GMM characteristics
                        enhancement_ratio = len(enhanced_features.columns) / len(features.columns)
                        enhancement_summary.append({
                            'specialist': specialist_name,
                            'original_features': len(features.columns),
                            'enhanced_features': len(enhanced_features.columns),
                            'enhancement_ratio': enhancement_ratio,
                            'samples': len(enhanced_features)
                        })
                        
                        tprint_success(f"✅ {specialist_name}: GMM enhanced {features.shape} → {enhanced_features.shape}")
                        tprint_info(f"   📈 Feature expansion: {len(features.columns)} → {len(enhanced_features.columns)} ({enhancement_ratio:.1f}x)")
                        tprint_info(f"   🎯 New feature types: {self._analyze_feature_types(enhanced_features)}")
                        tprint_info(f"   📊 Sample enhanced features: {list(enhanced_features.columns[:5])}")
                        
                        # Log GMM-specific features
                        gmm_features = [col for col in enhanced_features.columns if any(gmm_term in col.lower() for gmm_term in ['gmm', 'cluster', 'regime', 'state', 'probability', 'entropy', 'familiarity'])]
                        if gmm_features:
                            tprint_info(f"   🧠 GMM-specific features ({len(gmm_features)}): {gmm_features[:3]}..." if len(gmm_features) > 3 else f"   🧠 GMM-specific features: {gmm_features}")
                        
                    else:
                        tprint_warning(f"⚠️ {specialist_name}: GMM enhancement failed, using original features")
                        gmm_enhanced_specialists[specialist_name] = features
                        enhancement_summary.append({
                            'specialist': specialist_name,
                            'original_features': len(features.columns),
                            'enhanced_features': len(features.columns),
                            'enhancement_ratio': 1.0,
                            'samples': len(features),
                            'status': 'failed'
                        })
                        
                except Exception as e:
                    tprint_error(f"❌ {specialist_name}: GMM enhancement error: {e}")
                    # Use original features as fallback
                    gmm_enhanced_specialists[specialist_name] = features
                    enhancement_summary.append({
                        'specialist': specialist_name,
                        'original_features': len(features.columns),
                        'enhanced_features': len(features.columns),
                        'enhancement_ratio': 1.0,
                        'samples': len(features),
                        'status': 'error',
                        'error': str(e)
                    })
            
            if not gmm_enhanced_specialists:
                tprint_error("❌ No GMM-enhanced specialists created")
                return {}
            
            # Comprehensive enhancement summary
            tprint_success(f"✅ GMM enhancement completed for {len(gmm_enhanced_specialists)} specialists")
            
            # Log detailed summary table
            tprint_info("\n📊 GMM Enhancement Summary:")
            for summary in enhancement_summary:
                status_icon = "✅" if summary.get('status') != 'failed' and summary.get('status') != 'error' else "❌"
                tprint_info(f"   {status_icon} {summary['specialist']}: {summary['original_features']} → {summary['enhanced_features']} features ({summary['enhancement_ratio']:.1f}x)")
            
            # Log overall statistics
            total_original = sum(s['original_features'] for s in enhancement_summary)
            total_enhanced = sum(s['enhanced_features'] for s in enhancement_summary)
            successful_enhancements = sum(1 for s in enhancement_summary if s.get('status') not in ['failed', 'error'])
            
            tprint_info(f"\n📈 Overall Statistics:")
            tprint_info(f"   🎯 Total features: {total_original} → {total_enhanced} ({total_enhanced/total_original:.1f}x expansion)")
            tprint_info(f"   ✅ Successful enhancements: {successful_enhancements}/{len(enhancement_summary)}")
            tprint_info(f"   📊 Average expansion ratio: {total_enhanced/total_original:.1f}x")
            
            # Log feature type distribution
            all_enhanced_features = pd.concat(list(gmm_enhanced_specialists.values()), axis=1)
            feature_type_dist = self._analyze_feature_types(all_enhanced_features)
            tprint_info(f"\n🏷️ Feature Type Distribution: {feature_type_dist}")
            
            return gmm_enhanced_specialists
            
        except Exception as e:
            tprint_error(f"❌ Failed to apply GMM to specialists: {e}")
            return {}

    def _analyze_feature_types(self, features: pd.DataFrame) -> Dict[str, int]:
        """
        Analyze and categorize feature types in a DataFrame.
        
        Returns a dictionary with counts of different feature categories.
        """
        try:
            feature_categories = {
                'gmm': 0,
                'momentum': 0,
                'volatility': 0,
                'trend': 0,
                'volume': 0,
                'price': 0,
                'technical': 0,
                'regime': 0,
                'cluster': 0,
                'probability': 0,
                'entropy': 0,
                'familiarity': 0,
                'kinematics': 0,
                'other': 0
            }
            
            for col in features.columns:
                col_lower = col.lower()
                
                # Check for GMM-specific features
                if any(term in col_lower for term in ['gmm', 'gaussian', 'mixture']):
                    feature_categories['gmm'] += 1
                elif any(term in col_lower for term in ['cluster', 'state', 'regime']):
                    feature_categories['regime'] += 1
                elif any(term in col_lower for term in ['probability', 'prob']):
                    feature_categories['probability'] += 1
                elif any(term in col_lower for term in ['entropy', 'ent']):
                    feature_categories['entropy'] += 1
                elif any(term in col_lower for term in ['familiarity', 'fam']):
                    feature_categories['familiarity'] += 1
                elif any(term in col_lower for term in ['momentum', 'mom']):
                    feature_categories['momentum'] += 1
                elif any(term in col_lower for term in ['volatility', 'vol', 'atr']):
                    feature_categories['volatility'] += 1
                elif any(term in col_lower for term in ['trend', 'direction']):
                    feature_categories['trend'] += 1
                elif any(term in col_lower for term in ['volume', 'vol_', 'volatility']):
                    feature_categories['volume'] += 1
                elif any(term in col_lower for term in ['price', 'close', 'open', 'high', 'low']):
                    feature_categories['price'] += 1
                elif any(term in col_lower for term in ['kinematics', 'velocity', 'acceleration', 'jerk']):
                    feature_categories['kinematics'] += 1
                elif any(term in col_lower for term in ['rsi', 'macd', 'bb_', 'sma', 'ema', 'wma']):
                    feature_categories['technical'] += 1
                else:
                    feature_categories['other'] += 1
            
            # Return only non-zero categories
            return {k: v for k, v in feature_categories.items() if v > 0}
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to analyze feature types: {e}")
            return {'error': 1}

    async def _train_ridge_with_monotonic_constraints(
        self,
        features: pd.DataFrame,
        market_data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train Ridge model with monotonic constraints on selected GMM features.
        
        Ridge regression applies L2 regularization and can enforce monotonic relationships
        between features and target, which is useful for financial modeling.
        """
        try:
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score
            import numpy as np
            
            tprint_info(f"📐 Training Ridge model on {features.shape} features...")
            
            # Create target variable (forward returns)
            returns = market_data['close'].pct_change().shift(-12).fillna(0)
            
            # Align features and target
            aligned_features = features.reindex(returns.index).dropna()
            aligned_target = returns.reindex(aligned_features.index)
            
            if len(aligned_features) < 1000:
                tprint_warning(f"⚠️ Insufficient data for Ridge training: {len(aligned_features)} samples")
                return {"success": False, "error": "Insufficient data"}
            
            # Standardize features for Ridge
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(aligned_features)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Find optimal alpha through cross-validation
            alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            best_alpha = 1.0
            best_score = -np.inf
            
            tprint_info("🔍 Optimizing Ridge alpha parameter...")
            for alpha in alphas:
                ridge = Ridge(alpha=alpha, random_state=42)
                scores = cross_val_score(ridge, X_scaled, aligned_target, cv=tscv, scoring='neg_mean_squared_error')
                mean_score = scores.mean()
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha
            
            tprint_success(f"✅ Best Ridge alpha: {best_alpha} (CV Score: {-best_score:.6f})")
            
            # Train final model
            final_ridge = Ridge(alpha=best_alpha, random_state=42)
            final_ridge.fit(X_scaled, aligned_target)
            
            # Make predictions
            predictions = final_ridge.predict(X_scaled)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(aligned_target, predictions)
            r2 = r2_score(aligned_target, predictions)
            
            # Feature importance (coefficients)
            feature_importance = pd.DataFrame({
                'feature': aligned_features.columns,
                'coefficient': final_ridge.coef_,
                'abs_coefficient': np.abs(final_ridge.coef_)
            }).sort_values('abs_coefficient', ascending=False)
            
            # Identify monotonic relationships (positive coefficients)
            monotonic_features = feature_importance[feature_importance['coefficient'] > 0]['feature'].tolist()
            
            tprint_success(f"✅ Ridge model trained: MSE={mse:.6f}, R²={r2:.4f}")
            tprint_info(f"📈 Monotonic features: {len(monotonic_features)}/{len(aligned_features.columns)}")
            tprint_info(f"🎯 Top 5 Ridge features: {feature_importance['feature'].head(5).tolist()}")
            
            return {
                "success": True,
                "model": final_ridge,
                "scaler": scaler,
                "alpha": best_alpha,
                "mse": mse,
                "r2_score": r2,
                "feature_importance": feature_importance,
                "monotonic_features": monotonic_features,
                "predictions": predictions,
                "n_samples": len(aligned_features),
                "n_features": len(aligned_features.columns)
            }
            
        except Exception as e:
            tprint_error(f"❌ Ridge training failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _train_extratrees_on_gmm_features(
        self,
        features: pd.DataFrame,
        market_data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train ExtraTrees models on the selected GMM features.
        
        ExtraTrees (Extremely Randomized Trees) provide robust ensemble learning
        with good performance on financial data.
        """
        try:
            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score
            from sklearn.metrics import mean_squared_error, r2_score
            import numpy as np
            
            tprint_info(f"🌲 Training ExtraTrees on {features.shape} GMM features...")
            
            # Create target variable (forward returns)
            returns = market_data['close'].pct_change().shift(-12).fillna(0)
            
            # Align features and target
            aligned_features = features.reindex(returns.index).dropna()
            aligned_target = returns.reindex(aligned_features.index)
            
            if len(aligned_features) < 1000:
                tprint_warning(f"⚠️ Insufficient data for ExtraTrees training: {len(aligned_features)} samples")
                return {"success": False, "error": "Insufficient data"}
            
            # Time series cross-validation for hyperparameter tuning
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            best_params = {}
            best_score = -np.inf
            
            tprint_info("🔍 Optimizing ExtraTrees hyperparameters...")
            
            # Simple grid search
            for n_estimators in param_grid['n_estimators']:
                for max_depth in param_grid['max_depth']:
                    for min_samples_split in param_grid['min_samples_split']:
                        for min_samples_leaf in param_grid['min_samples_leaf']:
                            for max_features in param_grid['max_features']:
                                
                                et = ExtraTreesRegressor(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    max_features=max_features,
                                    random_state=42,
                                    n_jobs=-1
                                )
                                
                                try:
                                    scores = cross_val_score(et, aligned_features, aligned_target, 
                                                             cv=tscv, scoring='neg_mean_squared_error')
                                    mean_score = scores.mean()
                                    
                                    if mean_score > best_score:
                                        best_score = mean_score
                                        best_params = {
                                            'n_estimators': n_estimators,
                                            'max_depth': max_depth,
                                            'min_samples_split': min_samples_split,
                                            'min_samples_leaf': min_samples_leaf,
                                            'max_features': max_features
                                        }
                                except Exception:
                                    continue
            
            tprint_success(f"✅ Best ExtraTrees params: {best_params}")
            
            # Train final model
            final_et = ExtraTreesRegressor(
                **best_params,
                random_state=42,
                n_jobs=-1
            )
            final_et.fit(aligned_features, aligned_target)
            
            # Make predictions
            predictions = final_et.predict(aligned_features)
            
            # Calculate metrics
            mse = mean_squared_error(aligned_target, predictions)
            r2 = r2_score(aligned_target, predictions)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': aligned_features.columns,
                'importance': final_et.feature_importances_
            }).sort_values('importance', ascending=False)
            
            tprint_success(f"✅ ExtraTrees trained: MSE={mse:.6f}, R²={r2:.4f}")
            tprint_info(f"🎯 Top 5 ExtraTrees features: {feature_importance['feature'].head(5).tolist()}")
            
            return {
                "success": True,
                "model": final_et,
                "params": best_params,
                "mse": mse,
                "r2_score": r2,
                "feature_importance": feature_importance,
                "predictions": predictions,
                "n_samples": len(aligned_features),
                "n_features": len(aligned_features.columns)
            }
            
        except Exception as e:
            tprint_error(f"❌ ExtraTrees training failed: {e}")
            return {"success": False, "error": str(e)}
    async def _extract_raw_features_from_market_data(self, market_data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Extract raw features from market data without training specialists."""
        try:
            # Generate basic technical features from market data
            features = pd.DataFrame(index=market_data.index)
            
            # Price-based features
            features['returns'] = market_data['close'].pct_change()
            features['log_returns'] = np.log(market_data['close'] / market_data['close'].shift(1))
            features['high_low_ratio'] = market_data['high'] / market_data['low']
            features['volume_price_ratio'] = market_data['volume'] / market_data['close']
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                features[f'ma_{window}'] = market_data['close'].rolling(window).mean()
                features[f'ma_{window}_ratio'] = market_data['close'] / features[f'ma_{window}']
            
            # Volatility features
            for window in [5, 10, 20]:
                features[f'vol_{window}'] = features['returns'].rolling(window).std()
            
            # RSI
            delta = market_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # Drop NaN values
            features = features.dropna()
            
            tprint_info(f"📊 Generated {len(features.columns)} raw features from market data")
            return features
            
        except Exception as e:
            tprint_error(f"❌ Failed to extract raw features: {e}")
            return pd.DataFrame()
    
    async def _apply_feature_selection(self, features: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply feature selection to reduce dimensionality."""
        try:
            from sklearn.feature_selection import SelectKBest, f_regression
            
            # Create synthetic labels for feature selection (using returns)
            labels = features['returns'].shift(-1).dropna()
            features_clean = features.iloc[:-1].loc[labels.index]
            
            # Remove highly correlated features
            corr_matrix = features_clean.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            features_clean = features_clean.drop(columns=to_drop)
            
            # Select top features
            # Handle NaN values before SelectKBest
            features_clean = features_clean.fillna(0)  # Replace NaN with 0
            selector = SelectKBest(f_regression, k=min(50, len(features_clean.columns)))
            selected_features = selector.fit_transform(features_clean, labels)
            
            selected_df = pd.DataFrame(
                selected_features,
                index=features_clean.index,
                columns=features_clean.columns[selector.get_support()]
            )
            
            tprint_success(f"✅ Feature selection: {len(features.columns)} → {len(selected_df.columns)} features")
            return selected_df
            
        except Exception as e:
            tprint_error(f"❌ Feature selection failed: {e}")
            return features
    
    async def _train_models_on_selected_features(self, features: pd.DataFrame, market_data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train ExtraTrees, CatBoost, LGBM, and XGBoost models on selected features using Huber Teacher
        for feature pruning, monotonic constraints, and interaction constraints.
        """
        try:
            import optuna
            from optuna.pruners import MedianPruner
            from src.utils.huber_regressor_for_trees import prepare_huber_teacher_outputs
            from sklearn.ensemble import ExtraTreesRegressor
            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor
            from catboost import CatBoostRegressor, Pool
            from sklearn.metrics import roc_auc_score
            from scipy.stats import spearmanr

            tprint_info(f"🧠 Training ensemble models on {features.shape} features with Huber Teacher...")

            # 1. Prepare Target and Features
            # Target is forward returns (shift -12 as per original Ridge logic, or next bar?
            # Original Ridge used shift(-12). Feature selection used shift(-1).
            # The prompt says "IC to target * AUC". Usually target is next period return or N-period.
            # I will align with 'returns' column shift(-1) if available, or calc from market_data.
            # features['returns'] exists from raw extraction.
            # Using shift(-1) for standard prediction.

            if 'returns' in features.columns:
                target = features['returns'].shift(-1)
            else:
                 target = market_data['close'].pct_change().shift(-1)

            # Align features and target
            data = features.copy()
            data['target'] = target
            data = data.dropna()

            if len(data) < 100:
                 tprint_error("❌ Insufficient data for training")
                 return {}

            y = data['target']
            X = data.drop(columns=['target'])

            # 2. Split Data (70% Train, 30% OOF)
            split_idx = int(len(X) * 0.70)
            X_train, X_oof = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_oof = y.iloc[:split_idx], y.iloc[split_idx:]

            tprint_info(f"📊 Split: Train {len(X_train)}, OOF {len(X_oof)}")

            # 3. Huber Teacher (Feature Pruning, Monotonic, Interactions, Warm Start)
            tprint_info("👨‍🏫 Running Huber Teacher...")
            huber_outputs = prepare_huber_teacher_outputs(X_train, y_train, X_val=None, X_test=X_oof)

            selected_features_names = huber_outputs['selected_features']
            monotonic_constraints = huber_outputs['monotonic_constraints']
            interaction_constraints = huber_outputs['interaction_constraints']
            warm_start_train = huber_outputs['warm_start']['train']
            warm_start_oof = huber_outputs['warm_start']['test']

            # Filter X to selected features
            X_train_sel = X_train[selected_features_names]
            X_oof_sel = X_oof[selected_features_names]

            tprint_success(f"✅ Huber Pruning: {len(X.columns)} -> {len(selected_features_names)} features")
            
            # Convert monotonic_constraints dict to array for tree models that need it
            if isinstance(monotonic_constraints, dict):
                # Map selected features to their monotonic constraints
                mono_cst_array = []
                for feat in selected_features_names:
                    mono_cst_array.append(monotonic_constraints.get(feat, 0))
                mono_cst_array_np = np.array(mono_cst_array)  # numpy array for sklearn
                mono_cst_array_list = mono_cst_array_np.tolist()  # list for CatBoost
            else:
                mono_cst_array_np = np.array(monotonic_constraints) if not isinstance(monotonic_constraints, np.ndarray) else monotonic_constraints
                mono_cst_array_list = mono_cst_array_np.tolist()
            
            results = {}
            
            def calculate_score(y_true, y_pred):
                ic, _ = spearmanr(y_true, y_pred)
                auc = roc_auc_score((y_true > 0).astype(int), y_pred)
                return ic * auc, ic, auc

            # --- Model 1: ExtraTrees ---
            # Sklearn 1.4+ supports monotonic_cst
            tprint_info("🌲 Training ExtraTrees...")
            tprint_info(f"   🔧 Monotonic constraints: {mono_cst_array_np}")
            # Count constraint types
            neg_count = sum(1 for x in mono_cst_array_np if x == -1)
            pos_count = sum(1 for x in mono_cst_array_np if x == 1)
            zero_count = sum(1 for x in mono_cst_array_np if x == 0)
            tprint_info(f"   📊 Constraints: {neg_count} negative, {pos_count} positive, {zero_count} unconstrained")
            tprint_info(f"   📊 Feature count: {len(X_train_sel.columns)}")
            
            def objective_et(trial):
                n_estimators = trial.suggest_int('n_estimators', 100, 500)
                max_depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30])
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

                # Check for monotonic_cst support (sklearn >= 1.4)
                # If supported, use it. Else warn.
                try:
                    et = ExtraTreesRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        monotonic_cst=mono_cst_array_np, # Use numpy array for sklearn
                        n_jobs=-1,
                        random_state=42
                    )
                    et.fit(X_train_sel, y_train)
                except TypeError:
                     # Fallback for older sklearn
                     et = ExtraTreesRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        n_jobs=-1,
                        random_state=42
                    )
                     et.fit(X_train_sel, y_train)

                preds = et.predict(X_oof_sel)
                score, _, _ = calculate_score(y_oof, preds)
                return score

            study_et = optuna.create_study(direction='maximize', pruner=MedianPruner())
            study_et.optimize(objective_et, n_trials=10) # Limited trials

            # Retrain best ET
            best_et_params = study_et.best_params
            tprint_info(f"   🏆 Best ET params: {best_et_params}")
            try:
                best_et = ExtraTreesRegressor(**best_et_params, monotonic_cst=mono_cst_array_np, n_jobs=-1, random_state=42)
                tprint_info("   ✅ Using monotonic constraints")
            except TypeError:
                best_et = ExtraTreesRegressor(**best_et_params, n_jobs=-1, random_state=42)
                tprint_info("   ⚠️ Monotonic constraints not supported (older sklearn)")
            best_et.fit(X_train_sel, y_train)
            pred_et = best_et.predict(X_oof_sel)
            score_et, ic_et, auc_et = calculate_score(y_oof, pred_et)
            results['ExtraTrees'] = {'score': score_et, 'ic': ic_et, 'auc': auc_et, 'model': best_et}
            tprint_success(f"   ✅ ExtraTrees: Score={score_et:.4f}, IC={ic_et:.4f}, AUC={auc_et:.4f}")

            # --- Model 2: XGBoost ---
            tprint_info("🚀 Training XGBoost...")
            tprint_info(f"   🔧 Fixed params: {xgb_fixed_params}")
            tprint_info(f"   📊 Feature count: {len(X_train_sel.columns)}")
            # Fixed params from user
            xgb_fixed_params = {
                'num_parallel_tree': 7,
                'colsample_bynode': 0.4,
                'subsample': 0.6,
                'reg_lambda': 50, # "22 regularisation 50" -> lambda
                'min_child_weight': 10,
                'gamma': 1.1,
                'learning_rate': 0.03,
                'tree_method': 'hist',
                'n_jobs': -1,
                'monotone_constraints': monotonic_constraints, # tuple
                'interaction_constraints': interaction_constraints if interaction_constraints else None
            }
            
            def objective_xgb(trial):
                n_estimators = trial.suggest_int('n_estimators', 100, 1000)
                max_depth = trial.suggest_int('max_depth', 3, 10)

                model = XGBRegressor(**xgb_fixed_params, n_estimators=n_estimators, max_depth=max_depth)

                # Warm start using base_margin?
                # XGBoost uses base_margin for initial prediction.
                # However, sklearn API uses `fit(X, y, base_margin=...)`

                model.fit(X_train_sel, y_train, base_margin=warm_start_train, verbose=False)
                pred = model.predict(X_oof_sel, base_margin=warm_start_oof)

                score, _, _ = calculate_score(y_oof, pred)
                return score

            study_xgb = optuna.create_study(direction='maximize', pruner=MedianPruner())
            study_xgb.optimize(objective_xgb, n_trials=10)
            
            best_xgb_params = study_xgb.best_params
            tprint_info(f"   🏆 Best XGB params: {best_xgb_params}")
            best_xgb = XGBRegressor(**xgb_fixed_params, **best_xgb_params)
            best_xgb.fit(X_train_sel, y_train, base_margin=warm_start_train)
            pred_xgb = best_xgb.predict(X_oof_sel, base_margin=warm_start_oof)
            score_xgb, ic_xgb, auc_xgb = calculate_score(y_oof, pred_xgb)
            results['XGBoost'] = {'score': score_xgb, 'ic': ic_xgb, 'auc': auc_xgb, 'model': best_xgb}
            tprint_success(f"   ✅ XGBoost: Score={score_xgb:.4f}, IC={ic_xgb:.4f}, AUC={auc_xgb:.4f}")

            # --- Model 3: CatBoost ---
            tprint_info("🐱 Training CatBoost...")
            tprint_info(f"   🔧 Fixed params: {cb_fixed_params}")
            tprint_info(f"   📊 Feature count: {len(X_train_sel.columns)}")
            # Fixed params
            cb_fixed_params = {
                'subsample': 0.6,
                'colsample_bylevel': 0.5,
                'leaf_estimation_iterations': 10,
                'l2_leaf_reg': 20, # "l2_leaf_reg 20"
                'random_strength': 5,
                'bootstrap_type': 'MVS',
                'verbose': False,
                'allow_writing_files': False,
                # Monotonic constraints format in CatBoost: list/tuple of int
                'monotone_constraints': mono_cst_array_list
            }
            
            def objective_cb(trial):
                iterations = trial.suggest_int('iterations', 100, 1000)
                depth = trial.suggest_int('depth', 4, 10)

                # CatBoost warm start uses `baseline` argument in Pool or fit.
                train_pool = Pool(X_train_sel, y_train, baseline=warm_start_train)

                model = CatBoostRegressor(**cb_fixed_params, iterations=iterations, depth=depth)
                model.fit(train_pool)

                pred = model.predict(X_oof_sel) # No baseline for predict?
                # If trained with baseline, predict outputs raw + baseline?
                # CatBoost docs: "If baseline is provided... the model learns the correction..."
                # When predicting, we must provide baseline?
                # sklearn API predict() doesn't take baseline easily?
                # We can use eval_set with Pool?
                # Or just predict raw and add warm_start_oof?
                # Actually CatBoost `predict` doesn't strictly require baseline if we want raw leaf sum, but we want full prediction.
                # If we use `baseline` in fit, the model learns residual.
                # So Prediction = Baseline + Model(X).
                # We need to add baseline manually if predict doesn't take it?
                # `predict` takes `X`. `Pool` can take baseline.
                # Let's try passing Pool to predict?

                eval_pool = Pool(X_oof_sel, baseline=warm_start_oof)
                pred = model.predict(eval_pool)

                score, _, _ = calculate_score(y_oof, pred)
                return score

            study_cb = optuna.create_study(direction='maximize', pruner=MedianPruner())
            study_cb.optimize(objective_cb, n_trials=10)

            best_cb_params = study_cb.best_params
            tprint_info(f"   🏆 Best CB params: {best_cb_params}")
            best_cb = CatBoostRegressor(**cb_fixed_params, **best_cb_params)
            train_pool = Pool(X_train_sel, y_train, baseline=warm_start_train)
            best_cb.fit(train_pool)
            eval_pool = Pool(X_oof_sel, baseline=warm_start_oof)
            pred_cb = best_cb.predict(eval_pool)
            score_cb, ic_cb, auc_cb = calculate_score(y_oof, pred_cb)
            results['CatBoost'] = {'score': score_cb, 'ic': ic_cb, 'auc': auc_cb, 'model': best_cb}
            tprint_success(f"   ✅ CatBoost: Score={score_cb:.4f}, IC={ic_cb:.4f}, AUC={auc_cb:.4f}")

            # --- Model 4: LGBM ---
            tprint_info("🍃 Training LGBM...")
            tprint_info(f"   📊 Feature count: {len(X_train_sel.columns)}")
            tprint_info(f"   🔧 Monotonic constraints: {monotonic_constraints}")
            tprint_info(f"   🔧 Interaction constraints: {interaction_constraints}")
            # Optuna tuning + Interaction constraints

            def objective_lgbm(trial):
                n_estimators = trial.suggest_int('n_estimators', 100, 1000)
                num_leaves = trial.suggest_int('num_leaves', 20, 100)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)

                model = LGBMRegressor(
                    n_estimators=n_estimators,
                    num_leaves=num_leaves,
                    learning_rate=learning_rate,
                    monotone_constraints=monotonic_constraints,
                    interaction_constraints=interaction_constraints,
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1
                )

                # LGBM supports init_score
                model.fit(X_train_sel, y_train, init_score=warm_start_train)
                # Predict with init_score?
                # LGBMRegressor predict() supports `init_score`? No, usually not in sklearn API.
                # But if we trained on residuals (implied by init_score), the model predicts residuals.
                # So we must add warm_start_oof.
                raw_pred = model.predict(X_oof_sel)
                # raw_pred is correction.
                pred = raw_pred + warm_start_oof

                score, _, _ = calculate_score(y_oof, pred)
                return score

            study_lgbm = optuna.create_study(direction='maximize', pruner=MedianPruner())
            study_lgbm.optimize(objective_lgbm, n_trials=10)

            best_lgbm_params = study_lgbm.best_params
            tprint_info(f"   🏆 Best LGBM params: {best_lgbm_params}")
            best_lgbm = LGBMRegressor(
                **best_lgbm_params,
                monotone_constraints=monotonic_constraints,
                interaction_constraints=interaction_constraints,
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
            best_lgbm.fit(X_train_sel, y_train, init_score=warm_start_train)
            pred_lgbm_raw = best_lgbm.predict(X_oof_sel)
            pred_lgbm = pred_lgbm_raw + warm_start_oof
            score_lgbm, ic_lgbm, auc_lgbm = calculate_score(y_oof, pred_lgbm)
            results['LGBM'] = {'score': score_lgbm, 'ic': ic_lgbm, 'auc': auc_lgbm, 'model': best_lgbm}
            tprint_success(f"   ✅ LGBM: Score={score_lgbm:.4f}, IC={ic_lgbm:.4f}, AUC={auc_lgbm:.4f}")

            # 4. Compare and Pick Winner
            tprint_info("\n🏆 Model Comparison (Winner = IC * AUC on OOF):")
            best_model_name = None
            best_model_score = -float('inf')

            for name, res in results.items():
                tprint_info(f"   - {name}: Score={res['score']:.4f} (IC={res['ic']:.4f}, AUC={res['auc']:.4f})")
                if res['score'] > best_model_score:
                    best_model_score = res['score']
                    best_model_name = name

            tprint_success(f"🎉 Winner: {best_model_name} (Score: {best_model_score:.4f})")

            winner_result = results[best_model_name]

            return {
                "success": True,
                "winner_name": best_model_name,
                "winner_model": winner_result['model'],
                "winner_score": winner_result['score'],
                "winner_ic": winner_result['ic'],
                "winner_auc": winner_result['auc'],
                "all_results": {k: {'score': v['score'], 'ic': v['ic'], 'auc': v['auc']} for k,v in results.items()},
                "selected_features": selected_features_names
            }
            
        except Exception as e:
            tprint_error(f"❌ Ensemble training failed: {e}")
            import traceback
            tprint_error(traceback.format_exc())
            return {}

    async def _apply_gmm_enhancement_to_features(self, features: pd.DataFrame, market_data: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Apply comprehensive GMM enhancement using EnhancedGMMFeatures class."""
        tprint_info("🧠 Starting comprehensive GMM enhancement with EnhancedGMMFeatures...")
        
        # Import the enhanced GMM class
        from src.training.steps.market_analysis.gmm_enhanced_features import EnhancedGMMFeatures
        
        # Initialize EnhancedGMMFeatures with optimized configuration
        gmm_enhancer = EnhancedGMMFeatures(
            use_original_pipeline=True,
            use_enhanced_pipeline=True,
            use_fracdiff=True,
            use_treeshap=True,
            use_multi_timeframe=True,  # RE-ENABLED with optimizations
            use_streaming=True,
            n_clusters_macro=8,
            pca_variance=0.95,
            n_latent_factors=8,
            fracdiff_config={
                'max_d': 1.0,
                'min_d': 0.0,
                'adf_threshold': 0.01,
                'method': 'binary_search',
                'tolerance': 0.01
            },
            kinematics_config={
                'velocity_windows': [1, 3, 5],
                'acceleration_windows': [3, 5, 10],
                'jerk_windows': [5, 10, 15]
            },
            overextended_config={
                'return_threshold': 2.0,
                'entropy_threshold': 0.8,
                'z_familiarity_threshold': -2.0
            },
            shock_config={
                'probability_jump_threshold': 0.3,
                'z_familiarity_jump_threshold': 2.0,
                'entropy_drop_threshold': 0.2
            },
            treeshap_config={
                'n_estimators': 100,
                'max_depth': 8,
                'interaction_sample_size': 500,
                'importance_threshold': 0.01
            },
            multi_tf_config={
                'base_timeframe': "15m",
                'target_timeframes': ["15m", "60m", "4h"],
                'fusion_method': 'adaptive',
                'max_memory_mb': 2048,  # Increased for better performance
                'chunk_size': 50000,    # Increased for fewer iterations
                'use_entropy_bars': True  # Enable entropy-based alignment
            },
            verbose=True
        )
        
        # Prepare config for EnhancedGMMFeatures with market_data already loaded
        gmm_config = {
            'symbol': config.get('symbol', 'ETHUSDT'),
            'exchange': config.get('exchange', 'binance'),
            'timeframe': config.get('timeframe', '15m'),
            'direction': config.get('direction', 'long'),
            'execution_mode': 'full',
            'market_data': market_data  # Pass the already loaded market data
        }
        
        # Apply the full enhanced GMM pipeline
        tprint_info("🚀 Running full EnhancedGMMFeatures pipeline...")
        gmm_results = gmm_enhancer.run_with_data(gmm_config, market_data)
        
        # Extract enhanced features from results
        if gmm_results.get('success', False) and 'enhanced_features_path' in gmm_results:
            import pandas as pd
            enhanced_features_path = gmm_results['enhanced_features_path']
            if enhanced_features_path and os.path.exists(enhanced_features_path):
                enhanced_features = pd.read_parquet(enhanced_features_path)
                
                # Align indices with original features
                enhanced_features = enhanced_features.reindex(features.index, method='nearest')
                
                # Combine original features with enhanced GMM features
                combined_features = pd.concat([features, enhanced_features], axis=1)
                
                tprint_success(f"✅ Enhanced GMM pipeline completed: {features.shape} -> {combined_features.shape}")
                tprint_info(f"📊 Generated {combined_features.shape[1] - features.shape[1]} enhanced GMM features")
                
                return combined_features
            else:
                tprint_warning("⚠️ Enhanced GMM features file not found")
        
        tprint_warning("⚠️ Enhanced GMM pipeline failed")
        tprint_error("❌ Fast failing - GMM enhancement required for this pipeline")
        raise RuntimeError("Enhanced GMM pipeline failed - cannot proceed without GMM features")
    
    def _apply_basic_gmm_fallback(self, features: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Basic GMM fallback implementation."""
        try:
            from sklearn.mixture import GaussianMixture
            
            # Prepare data for GMM
            clean_features = features.dropna()
            if len(clean_features) < 100:
                tprint_warning("⚠️ Insufficient data for GMM enhancement")
                return features
            
            # Apply basic GMM clustering
            n_components = min(8, len(clean_features.columns))
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(clean_features)
            
            # Generate basic GMM features
            gmm_probs = gmm.predict_proba(clean_features)
            gmm_labels = gmm.predict(clean_features)
            
            # Create enhanced features
            enhanced_features = clean_features.copy()
            for i in range(n_components):
                enhanced_features[f'gmm_prob_{i}'] = gmm_probs[:, i]
                enhanced_features[f'gmm_cluster_{i}'] = (gmm_labels == i).astype(int)
            
            # Add entropy feature
            entropy = -np.sum(gmm_probs * np.log(gmm_probs + 1e-10), axis=1)
            enhanced_features['gmm_entropy'] = entropy
            
            tprint_success(f"✅ Basic GMM fallback: {features.shape} -> {enhanced_features.shape}")
            
            return enhanced_features
            
        except Exception as e:
            tprint_error(f"❌ Basic GMM fallback failed: {e}")
            return features
    

    async def _extract_specialist_features(
        self,
        specialist_outputs: Dict[str, pd.DataFrame],
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Extract ALL features from specialist outputs (except prediction/probability columns).
        
        This method takes all columns from each specialist except for
        prediction/probability/target columns, ensuring we capture the complete
        feature space for GMM processing.
        """
        try:
            tprint_info(f"🔍 Extracting ALL features from {len(specialist_outputs)} specialists...")
            
            specialist_features = {}
            
            for specialist_name, outputs in specialist_outputs.items():
                if outputs is not None and not outputs.empty:
                    # Check if this is a DataFrame with feature data
                    if isinstance(outputs, pd.DataFrame):
                        # Exclude only prediction/probability/target columns, keep everything else
                        exclude_cols = ['prediction', 'probability', 'target', 'label', 'timestamp']
                        feature_cols = [col for col in outputs.columns if col not in exclude_cols]
                        
                        if feature_cols:
                            specialist_features[specialist_name] = outputs[feature_cols]
                            tprint_info(f"✅ {specialist_name}: {len(feature_cols)} features extracted")
                            tprint_info(f"   Sample features: {feature_cols[:5]}" if len(feature_cols) > 5 else f"   Features: {feature_cols}")
                        else:
                            tprint_warning(f"⚠️ {specialist_name}: No features found (all columns excluded)")
                    else:
                        tprint_warning(f"⚠️ {specialist_name}: Output is not a DataFrame")
                else:
                    tprint_warning(f"⚠️ {specialist_name}: Empty or None output")
            
            if not specialist_features:
                tprint_error("❌ No specialist features extracted")
                return pd.DataFrame()
            
            # Combine all specialist features
            tprint_info("🔗 Combining specialist features...")
            combined_features = pd.concat(
                [features.add_prefix(f"{name}_") for name, features in specialist_features.items()],
                axis=1
            )
            
            tprint_success(f"✅ Combined features shape: {combined_features.shape}")
            tprint_info(f"📊 Feature breakdown: {', '.join([f'{name}: {len(features)}' for name, features in specialist_features.items()])}")
            
            return combined_features
            
        except Exception as e:
            tprint_error(f"❌ Failed to extract specialist features: {e}")
            return pd.DataFrame()
    async def _train_all_specialists(
        self,
        market_data: pd.DataFrame,
        config: Dict[str, Any],
        force_retrain: bool,
        batch_size: int = 3
    ) -> Dict[str, pd.DataFrame]:
        """Train all specialist models in memory-efficient batches."""

        tprint_info(f"🔄 Training specialists in batches of {batch_size}...")

        specialist_outputs = {}
        failed_specialists = []

        # Process specialists in batches
        specialist_items = list(SPECIALIST_IMPORTS.items())

        for i in range(0, len(specialist_items), batch_size):
            batch = specialist_items[i:i + batch_size]
            batch_names = [name for name, _ in batch]

            tprint_info(f"📦 Processing batch {i//batch_size + 1}: {batch_names}")

            # Memory snapshot before batch
            import psutil
            mem_b = psutil.Process().memory_info().rss / 1024 / 1024
            tprint_info(f"💾 Memory usage before batch: {mem_b:.1f} MB")

            # Train batch in parallel
            batch_outputs = await self._train_specialist_batch(
                batch, market_data, config, force_retrain
            )

            # Memory snapshot after batch
            mem_a = psutil.Process().memory_info().rss / 1024 / 1024
            tprint_info(f"💾 Memory usage after batch: {mem_a:.1f} MB (Delta: {mem_a - mem_b:+.1f} MB)")

            # Store successful outputs
            for step_name, outputs in batch_outputs.items():
                if outputs is not None and not outputs.empty:
                    specialist_outputs[step_name] = outputs
                else:
                    failed_specialists.append(step_name)

            # Force garbage collection between batches
            gc.collect()
            
            # Increment batch count and optimize memory periodically
            self.batch_count += 1
            if self.batch_count % self.memory_config.gc_frequency == 0:
                self._optimize_memory_usage()

            # Aggressive memory cleanup between batches
            gc.collect()
            # Force cleanup of any cached data
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            tprint_info(".1f")

            # Small delay to prevent overwhelming the system
            await asyncio.sleep(1.0)

        if failed_specialists:
            tprint_warning(f"⚠️ Failed specialists: {failed_specialists}")

        tprint_success(f"✅ Completed training {len(specialist_outputs)} specialists in {len(specialist_items)//batch_size + 1} batches")

        return specialist_outputs

    async def _train_all_specialists_memory_efficient(
        self,
        market_data: pd.DataFrame,
        config: Dict[str, Any],
        force_retrain: bool,
        batch_size: int = 1
    ) -> Dict[str, pd.DataFrame]:
        """Train all specialist models with memory-efficient disk-based storage."""
        import tempfile
        import pickle

        # #region agent log - Specialist training start
        with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "log_specialist_training_start",
                "timestamp": int(__import__('time').time() * 1000),
                "location": "train_specialists_with_gmm_step.py:_train_all_specialists_memory_efficient",
                "message": "Starting specialist training",
                "data": {"batch_size": batch_size, "market_data_shape": market_data.shape, "specialist_count": len(SPECIALIST_IMPORTS)},
                "sessionId": "debug-session",
                "runId": "initial",
                "hypothesisId": "H"
            }) + '\n')
        # #endregion

        tprint_info(f"🔄 Training specialists in memory-efficient mode (batch_size={batch_size})...")

        specialist_outputs = {}
        temp_dir = Path(tempfile.mkdtemp(prefix="specialist_outputs_"))

        try:
            # Process specialists in batches
            specialist_items = list(SPECIALIST_IMPORTS.items())

            for i in range(0, len(specialist_items), batch_size):
                batch = specialist_items[i:i + batch_size]
                batch_names = [name for name, _ in batch]

                # #region agent log - Batch processing start
                with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": f"log_batch_start_{i//batch_size + 1}",
                        "timestamp": int(__import__('time').time() * 1000),
                        "location": "train_specialists_with_gmm_step.py:_train_all_specialists_memory_efficient",
                        "message": f"Starting batch {i//batch_size + 1}",
                        "data": {"batch_number": i//batch_size + 1, "batch_names": batch_names, "total_batches": len(specialist_items)//batch_size + 1},
                        "sessionId": "debug-session",
                        "runId": "initial",
                        "hypothesisId": "H"
                    }) + '\n')
                # #endregion

                tprint_info(f"📦 Processing batch {i//batch_size + 1}/{len(specialist_items)//batch_size + 1}: {batch_names}")

                # Train batch
                batch_outputs = await self._train_specialist_batch(
                    batch, market_data, config, force_retrain
                )

                # #region agent log - Batch processing complete
                with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": f"log_batch_complete_{i//batch_size + 1}",
                        "timestamp": int(__import__('time').time() * 1000),
                        "location": "train_specialists_with_gmm_step.py:_train_all_specialists_memory_efficient",
                        "message": f"Completed batch {i//batch_size + 1}",
                        "data": {"batch_number": i//batch_size + 1, "batch_outputs_count": len(batch_outputs), "successful_outputs": len([k for k, v in batch_outputs.items() if v is not None and not v.empty])},
                        "sessionId": "debug-session",
                        "runId": "initial",
                        "hypothesisId": "H"
                    }) + '\n')
                # #endregion

                # Save successful outputs to disk immediately
                for step_name, outputs in batch_outputs.items():
                    if outputs is not None and not outputs.empty:
                        # Save to temporary file
                        temp_file = temp_dir / f"{step_name}.pkl"
                        with open(temp_file, 'wb') as f:
                            pickle.dump(outputs, f)
                        specialist_outputs[step_name] = temp_file  # Store file path instead of data
                        tprint_success(f"✅ {step_name} saved to disk ({outputs.shape})")
                    else:
                        tprint_warning(f"⚠️ {step_name} produced no outputs")

                # Aggressive memory cleanup between batches
                del batch_outputs
                gc.collect()
                
                # Trigger memory optimization (clears caches)
                self._optimize_memory_usage()

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(1.0)

            # Convert file paths back to loaded data for GMM processing
            loaded_outputs = {}
            for step_name, file_path in specialist_outputs.items():
                try:
                    with open(file_path, 'rb') as f:
                        loaded_outputs[step_name] = pickle.load(f)
                    tprint_info(f"📥 Loaded {step_name} from disk")
                except Exception as e:
                    tprint_error(f"❌ Failed to load {step_name} from disk: {e}")

            if loaded_outputs:
                tprint_success(f"✅ Loaded {len(loaded_outputs)} specialist outputs for GMM processing")

            return loaded_outputs

        finally:
            # Cleanup temporary files
            import shutil
            try:
                shutil.rmtree(temp_dir)
                tprint_info("🧹 Cleaned up temporary files")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to cleanup temp files: {e}")

    async def _train_specialist_batch(
        self,
        batch: List[Tuple[str, Tuple[str, str]]],
        market_data: pd.DataFrame,
        config: Dict[str, Any],
        force_retrain: bool
    ) -> Dict[str, pd.DataFrame]:
        """Train a batch of specialists in parallel."""

        # #region agent log - Batch training start
        with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "log_batch_train_start",
                "timestamp": int(__import__('time').time() * 1000),
                "location": "train_specialists_with_gmm_step.py:_train_specialist_batch",
                "message": "Starting batch training",
                "data": {"batch_size": len(batch), "specialist_names": [name for name, _ in batch]},
                "sessionId": "debug-session",
                "runId": "initial",
                "hypothesisId": "H"
            }) + '\n')
        # #endregion

        batch_outputs = {}

        # Create tasks for parallel execution
        tasks = []
        for step_name, (module_path, class_name) in batch:
            task = self._train_single_specialist_async(
                step_name, module_path, class_name, market_data, config, force_retrain
            )
            tasks.append(task)

        # #region agent log - Before asyncio.gather
        with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "log_before_gather",
                "timestamp": int(__import__('time').time() * 1000),
                "location": "train_specialists_with_gmm_step.py:_train_specialist_batch",
                "message": "Before asyncio.gather",
                "data": {"task_count": len(tasks)},
                "sessionId": "debug-session",
                "runId": "initial",
                "hypothesisId": "H"
            }) + '\n')
        # #endregion

        # Execute batch in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # #region agent log - After asyncio.gather
        with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "log_after_gather",
                "timestamp": int(__import__('time').time() * 1000),
                "location": "train_specialists_with_gmm_step.py:_train_specialist_batch",
                "message": "After asyncio.gather",
                "data": {"result_count": len(results)},
                "sessionId": "debug-session",
                "runId": "initial",
                "hypothesisId": "H"
            }) + '\n')
        # #endregion

        # Process results
        for i, result in enumerate(results):
            step_name = batch[i][0]

            if isinstance(result, Exception):
                tprint_error(f"❌ {step_name} failed: {result}")
                batch_outputs[step_name] = None
            else:
                batch_outputs[step_name] = result

                if result is not None and not result.empty:
                    tprint_success(f"✅ {step_name} completed, outputs: {result.shape}")
                else:
                    tprint_warning(f"⚠️ {step_name} produced no outputs")

        return batch_outputs

    async def _train_single_specialist_async(
        self,
        step_name: str,
        module_path: str,
        class_name: str,
        market_data: pd.DataFrame,
        config: Dict[str, Any],
        force_retrain: bool
    ) -> Optional[pd.DataFrame]:
        """Train a single specialist asynchronously."""

        # #region agent log - Single specialist start
        with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": f"log_single_specialist_start_{step_name}",
                "timestamp": int(__import__('time').time() * 1000),
                "location": "train_specialists_with_gmm_step.py:_train_single_specialist_async",
                "message": f"Starting training for specialist: {step_name}",
                "data": {"step_name": step_name, "module_path": module_path, "class_name": class_name},
                "sessionId": "debug-session",
                "runId": "initial",
                "hypothesisId": "H"
            }) + '\n')
        # #endregion

        try:
            # #region agent log - Entered try block
            with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "id": f"log_entered_try_{step_name}",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "train_specialists_with_gmm_step.py:_train_single_specialist_async",
                    "message": f"Entered try block for specialist: {step_name}",
                    "data": {"step_name": step_name},
                    "sessionId": "debug-session",
                    "runId": "initial",
                    "hypothesisId": "H"
                }) + '\n')
            # #endregion

            # Debug print - check if function continues
            print(f"DEBUG: About to enter try block for specialist: {step_name}")

            # #region agent log - Before try block
            try:
                with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": f"log_before_try_{step_name}",
                        "timestamp": int(__import__('time').time() * 1000),
                        "location": "train_specialists_with_gmm_step.py:_train_single_specialist_async",
                        "message": f"About to enter try block for specialist: {step_name}",
                        "data": {"step_name": step_name},
                        "sessionId": "debug-session",
                        "runId": "initial",
                        "hypothesisId": "H"
                    }) + '\n')
            except Exception as log_e:
                print(f"DEBUG: Failed to write log: {log_e}")
            # #endregion

            start_time = time.time()

            # Import specialist class
            # #region agent log - Import specialist class
            with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "id": f"log_import_specialist_{step_name}",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "train_specialists_with_gmm_step.py:_train_single_specialist_async",
                    "message": f"Importing specialist class: {module_path}.{class_name}",
                    "data": {"step_name": step_name, "module_path": module_path, "class_name": class_name},
                    "sessionId": "debug-session",
                    "runId": "initial",
                    "hypothesisId": "H"
                }) + '\n')
            # #endregion

            module = __import__(module_path, fromlist=[class_name])
            specialist_class = getattr(module, class_name)

            # Initialize specialist
            # #region agent log - Specialist initialization
            with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "id": f"log_specialist_init_{step_name}",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "train_specialists_with_gmm_step.py:_train_single_specialist_async",
                    "message": f"Initializing specialist: {step_name}",
                    "data": {"step_name": step_name, "class_name": class_name},
                    "sessionId": "debug-session",
                    "runId": "initial",
                    "hypothesisId": "H"
                }) + '\n')
            # #endregion

            specialist = specialist_class(step_name)

            # #region agent log - Specialist initialized successfully
            with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "id": f"log_specialist_init_success_{step_name}",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "train_specialists_with_gmm_step.py:_train_single_specialist_async",
                    "message": f"Specialist initialized successfully: {step_name}",
                    "data": {"step_name": step_name, "specialist_type": str(type(specialist))},
                    "sessionId": "debug-session",
                    "runId": "initial",
                    "hypothesisId": "H"
                }) + '\n')
            # #endregion

            # Setup context - enhanced specialists need proper context before execution
            context = {
                "symbol": config.get("symbol"),
                "exchange": config.get("exchange"),
                "timeframe": config.get("timeframe"),
                "direction": config.get("direction"),
                "model": "analyst"
            }

            # Set context on specialist if it has the attribute
            if hasattr(specialist, '_current_context'):
                specialist._current_context = context

            # Train specialist and get outputs
            outputs = await self._train_single_specialist(
                specialist, market_data, context, force_retrain, step_name
            )

            elapsed = time.time() - start_time
            if outputs is not None and not outputs.empty:
                tprint_info(f"⏱️ {step_name} trained in {elapsed:.2f}s")

            return outputs

        except Exception as e:
            tprint_error(f"❌ {step_name} training failed: {e}")
            return None

    async def _train_single_specialist(
        self,
        specialist,
        market_data: pd.DataFrame,
        context: Dict[str, Any],
        force_retrain: bool,
        step_name: str
    ) -> Optional[pd.DataFrame]:
        """Train a single specialist and return its outputs."""

        try:
            # Set context
            if hasattr(specialist, '_current_context'):
                specialist._current_context = context

            # Execute specialist training
            specialist_config = context.copy()
            specialist_config.update({
                "force_retrain": force_retrain,
                "verbose": False,  # Reduce verbosity for batch training
                "market_data": market_data  # Pass market data directly to avoid artifact loading
            })

            # #region agent log - Before specialist execute
            with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "id": f"log_before_execute_{step_name}",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "train_specialists_with_gmm_step.py:_train_single_specialist_async",
                    "message": f"Before executing specialist: {step_name}",
                    "data": {"step_name": step_name, "specialist_config_keys": list(specialist_config.keys())},
                    "sessionId": "debug-session",
                    "runId": "initial",
                    "hypothesisId": "H"
                }) + '\n')
            # #endregion

            # Run specialist
            try:
                # #region agent log - About to call execute
                with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": f"log_about_to_execute_{step_name}",
                        "timestamp": int(__import__('time').time() * 1000),
                        "location": "train_specialists_with_gmm_step.py:_train_single_specialist_async",
                        "message": f"About to call execute on specialist: {step_name}",
                        "data": {"step_name": step_name, "config_keys": list(specialist_config.keys())},
                        "sessionId": "debug-session",
                        "runId": "initial",
                        "hypothesisId": "H"
                    }) + '\n')
                # #endregion

                result = await specialist.execute(specialist_config)

                # #region agent log - After specialist execute
                with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": f"log_after_execute_{step_name}",
                        "timestamp": int(__import__('time').time() * 1000),
                        "location": "train_specialists_with_gmm_step.py:_train_single_specialist_async",
                        "message": f"Specialist executed: {step_name}",
                        "data": {"step_name": step_name, "result_type": type(result).__name__, "result_success": result.get('success', False) if isinstance(result, dict) else None, "result_keys": list(result.keys()) if isinstance(result, dict) else None, "result_is_none": result is None},
                        "sessionId": "debug-session",
                        "runId": "initial",
                        "hypothesisId": "H"
                    }) + '\n')
                # #endregion
            except Exception as e:
                # #region agent log - Specialist execution failed
                with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": f"log_execute_failed_{step_name}",
                        "timestamp": int(__import__('time').time() * 1000),
                        "location": "train_specialists_with_gmm_step.py:_train_single_specialist_async",
                        "message": f"Specialist execution failed: {step_name}",
                        "data": {"step_name": step_name, "error": str(e), "error_type": type(e).__name__},
                        "sessionId": "debug-session",
                        "runId": "initial",
                        "hypothesisId": "H"
                    }) + '\n')
                # #endregion
                tprint_error(f"❌ Specialist {step_name} failed: {e}")
                return None

            # Extract outputs/predictions from result and collect comprehensive metrics
            if result and isinstance(result, dict):
                # Store comprehensive metrics for this specialist
                specialist_name = specialist.__class__.__name__
                self._specialist_metrics[specialist_name] = {
                    'auc': result.get('metrics', {}).get('auc', 0.0),
                    'mi_score': result.get('metrics', {}).get('mi_score', 0.0),
                    'best_auc': result.get('metrics', {}).get('best_auc', 0.0),
                    'best_mi': result.get('metrics', {}).get('best_mi', 0.0),
                    'n_features': result.get('metrics', {}).get('n_features', 0),
                    'hyperparams': result.get('metrics', {}).get('optimization_params', {}),
                    'diagnostics': result.get('diagnostics', {}),
                    'compliance': result.get('metrics', {}).get('enhanced_requirements_met', False),
                    'enhanced_mi_score': result.get('metrics', {}).get('enhanced_mi_score', 0.0),
                    'ensemble_ready': result.get('metrics', {}).get('ensemble_ready', False),
                    'training_success': result.get('success', False),
                    'n_samples': result.get('n_samples', 0)
                }

                # Look for common output keys
                outputs = None
                for key in ["predictions", "features", "outputs", "probabilities"]:
                    if key in result and result[key] is not None:
                        outputs = result[key]
                        if isinstance(outputs, dict):
                            # Convert dict to DataFrame
                            outputs = pd.DataFrame(outputs)
                        break

                if outputs is not None:
                    # Ensure DatetimeIndex
                    if not isinstance(outputs.index, pd.DatetimeIndex):
                        if "timestamp" in outputs.columns:
                            outputs.index = pd.to_datetime(outputs["timestamp"])
                            outputs = outputs.drop(columns=["timestamp"])
                        else:
                            # Use market_data index
                            outputs.index = market_data.index[:len(outputs)]

                    # Store outputs for later reference
                    self._specialist_outputs[specialist_name] = outputs
                    return outputs

            return None

        except Exception as e:
            tprint_error(f"❌ Specialist training failed: {e}")
            return None

        finally:
            # Memory cleanup after each specialist
            await self._cleanup_specialist_memory(specialist)

    async def _cleanup_specialist_memory(self, specialist) -> None:
        """Clean up memory after specialist training to prevent memory bloat."""
        try:
            # Clear specialist caches
            if hasattr(specialist, '_market_data_cache'):
                specialist._market_data_cache.clear()

            if hasattr(specialist, 'feature_pipeline'):
                # Clear any cached data in feature pipeline
                if hasattr(specialist.feature_pipeline, 'cache'):
                    specialist.feature_pipeline.cache.clear()
                if hasattr(specialist.feature_pipeline, 'memory_cache'):
                    specialist.feature_pipeline.memory_cache.clear()

            # Clear training artifacts if they exist
            if hasattr(specialist, 'training_metrics'):
                specialist.training_metrics.clear()

            if hasattr(specialist, 'mi_history'):
                specialist.mi_history.clear()

            # Force garbage collection
            import gc
            gc.collect()

            # Clear any numpy/pandas temporary data
            if hasattr(specialist, '_temp_dataframes'):
                specialist._temp_dataframes.clear()

        except Exception as e:
            # Don't let cleanup failures interrupt training
            tprint_warning(f"⚠️ Memory cleanup warning (non-critical): {e}")
            pass

    def _combine_specialist_outputs(self, specialist_outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Memory-efficient combination of specialist outputs with sparse optimization."""

        if not specialist_outputs:
            return pd.DataFrame()

        try:
            tprint_info("🔄 Memory-efficient combination of specialist outputs...")
            
            # Pre-calculate total columns and validate indices
            total_columns = 0
            common_index = None
            sparse_features = {}
            
            for specialist_name, outputs in specialist_outputs.items():
                if outputs is not None and not outputs.empty:
                    total_columns += len(outputs.columns)
                    
                    # Check for sparse features
                    for col in outputs.columns:
                        sparsity = (outputs[col] == 0).mean()
                        if sparsity > self.memory_config.sparse_threshold:
                            sparse_features[f"{specialist_name}_{col}"] = outputs[col]
                    
                    # Use first valid DataFrame as index reference
                    if common_index is None:
                        common_index = outputs.index
                    # Check index compatibility
                    elif not outputs.index.equals(common_index):
                        tprint_warning(f"⚠️ Index mismatch for {specialist_name}, attempting to align...")
                        try:
                            # Try to align with common_index
                            outputs = outputs.reindex(common_index)
                        except Exception as e:
                            tprint_warning(f"⚠️ Could not align {specialist_name}: {e}")
                            continue
            
            if common_index is None:
                tprint_warning("⚠️ No valid specialist outputs found")
                return pd.DataFrame()
            
            # Use memory pool for combined data
            combined_data = {}
            
            # Process each specialist output without copying
            for specialist_name, outputs in specialist_outputs.items():
                if outputs is not None and not outputs.empty:
                    try:
                        # Ensure index alignment without copying
                        if not outputs.index.equals(common_index):
                            outputs = outputs.reindex(common_index)
                        
                        # Add prefixed columns directly to data dictionary
                        for col in outputs.columns:
                            feature_name = f"{specialist_name}_{col}"
                            
                            # Use sparse format for sparse features
                            if feature_name in sparse_features:
                                combined_data[feature_name] = sparse_features[feature_name]
                            else:
                                combined_data[feature_name] = outputs[col]
                            
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to process {specialist_name}: {e}")
                        continue
            
            if not combined_data:
                tprint_warning("⚠️ No valid data to combine")
                return pd.DataFrame()
            
            # Create DataFrame in one operation (no intermediate copies)
            combined = pd.DataFrame(combined_data, index=common_index)
            
            # Memory optimization: convert to appropriate dtypes
            for col in combined.columns:
                if combined[col].dtype == 'float64':
                    # Try to downcast to float32 if precision allows
                    try:
                        combined[col] = pd.to_numeric(combined[col], downcast='float')
                    except Exception:
                        pass
                elif combined[col].dtype == 'int64':
                    # Try to downcast integers
                    try:
                        combined[col] = pd.to_numeric(combined[col], downcast='integer')
                    except Exception:
                        pass
            
            # Force garbage collection to free memory
            gc.collect()
            
            tprint_success(f"✅ Combined {len(specialist_outputs)} specialists into {combined.shape} DataFrame")
            tprint_info(f"📊 Memory usage: {combined.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            if sparse_features:
                tprint_info(f"🎯 Sparse optimization: {len(sparse_features)} sparse features identified")
            
            return combined
            
        except Exception as e:
            tprint_error(f"❌ Memory-efficient combination failed: {e}")
            # Fallback to original method
            return self._combine_specialist_outputs_fallback(specialist_outputs)

    def _combine_specialist_outputs_fallback(self, specialist_outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Fallback method for combining specialist outputs."""

        all_features = []

        for specialist_name, outputs in specialist_outputs.items():
            if outputs is not None and not outputs.empty:
                # Prefix column names with specialist name
                prefixed_outputs = outputs.copy()
                prefixed_outputs.columns = [f"{specialist_name}_{col}" for col in outputs.columns]
                all_features.append(prefixed_outputs)

        if all_features:
            # Concatenate all features, aligning on index
            combined = pd.concat(all_features, axis=1, join='outer')
            return combined
        else:
            return pd.DataFrame()

    # Numba-optimized functions for vectorized feature selection
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def _vectorized_mutual_info_batch(feature_data: np.ndarray, target_data: np.ndarray, n_bins: int = 50) -> np.ndarray:
        """Calculate mutual information for multiple features in parallel using Numba."""
        n_features = feature_data.shape[1]
        mi_scores = np.zeros(n_features)
        
        # Discretize data using binning
        feature_min = feature_data.min()
        feature_max = feature_data.max()
        target_min = target_data.min()
        target_max = target_data.max()
        
        feature_bins = np.linspace(feature_min, feature_max, n_bins + 1)
        target_bins = np.linspace(target_min, target_max, n_bins + 1)
        
        for i in prange(n_features):
            try:
                # Discretize feature and target
                feature_discrete = np.digitize(feature_data[:, i], feature_bins) - 1
                target_discrete = np.digitize(target_data, target_bins) - 1
                
                # Clip to valid range
                feature_discrete = np.clip(feature_discrete, 0, n_bins - 1)
                target_discrete = np.clip(target_discrete, 0, n_bins - 1)
                
                # Calculate joint histogram manually
                joint_hist = np.zeros((n_bins, n_bins))
                for j in range(len(feature_discrete)):
                    joint_hist[feature_discrete[j], target_discrete[j]] += 1
                
                # Calculate total count
                total_count = len(feature_discrete)
                if total_count == 0:
                    mi_scores[i] = 0.0
                    continue
                
                # Calculate mutual information manually
                mi = 0.0
                for fi in range(n_bins):
                    for ti in range(n_bins):
                        joint_count = joint_hist[fi, ti]
                        if joint_count > 0:
                            # Calculate marginal counts
                            feature_count = 0
                            for t in range(n_bins):
                                feature_count += joint_hist[fi, t]
                            
                            target_count = 0
                            for f in range(n_bins):
                                target_count += joint_hist[f, ti]
                            
                            if feature_count > 0 and target_count > 0:
                                joint_prob = joint_count / total_count
                                feature_prob = feature_count / total_count
                                target_prob = target_count / total_count
                                
                                mi += joint_prob * np.log(joint_prob / (feature_prob * target_prob))
                
                mi_scores[i] = mi
                
            except Exception:
                mi_scores[i] = 0.0
        
        return mi_scores

    def _check_feature_quality(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Check feature quality by detecting constant features and features with insufficient variance.
        
        Args:
            features: DataFrame containing features to check
            
        Returns:
            Tuple containing (cleaned_features, quality_report)
        """
        from typing import Tuple, Dict, Any
        
        quality_report = {
            'total_features': len(features.columns),
            'constant_features': [],
            'low_variance_features': [],
            'kept_features': [],
            'variance_threshold': 1e-8
        }
        
        cleaned_features = features.copy()
        
        # Check for constant features
        for col in features.columns:
            std_val = features[col].std()
            
            if std_val == 0:
                quality_report['constant_features'].append(col)
                cleaned_features.drop(col, axis=1, inplace=True)
            elif std_val < quality_report['variance_threshold']:
                quality_report['low_variance_features'].append({
                    'feature': col,
                    'variance': std_val
                })
            else:
                quality_report['kept_features'].append(col)
        
        quality_report['final_feature_count'] = len(cleaned_features.columns)
        quality_report['removed_features'] = quality_report['total_features'] - quality_report['final_feature_count']
        
        return cleaned_features, quality_report

    def _list_gmm_features(self, features: pd.DataFrame, source: str = "unknown") -> Dict[str, Any]:
        """
        List GMM features with information about them.
        
        Args:
            features: DataFrame containing GMM features
            source: Source of the features (e.g., "returns_pipeline", "fracdiff_pipeline")
            
        Returns:
            Dictionary containing feature information
        """
        from typing import Dict, Any
        
        feature_info = {
            'source': source,
            'total_features': len(features.columns),
            'feature_details': [],
            'statistics': {
                'mean_variance': 0.0,
                'min_variance': float('inf'),
                'max_variance': 0.0,
                'constant_features': 0,
                'low_variance_features': 0
            }
        }
        
        for col in features.columns:
            feature_data = features[col]
            variance = feature_data.var()
            mean = feature_data.mean()
            std = feature_data.std()
            
            feature_details = {
                'feature_name': col,
                'mean': mean,
                'variance': variance,
                'std_deviation': std,
                'is_constant': variance == 0,
                'is_low_variance': variance < 1e-8
            }
            
            feature_info['feature_details'].append(feature_details)
            
            # Update statistics
            feature_info['statistics']['mean_variance'] += variance
            if variance < feature_info['statistics']['min_variance']:
                feature_info['statistics']['min_variance'] = variance
            if variance > feature_info['statistics']['max_variance']:
                feature_info['statistics']['max_variance'] = variance
            
            if variance == 0:
                feature_info['statistics']['constant_features'] += 1
            elif variance < 1e-8:
                feature_info['statistics']['low_variance_features'] += 1
        
        # Calculate mean variance
        if feature_info['total_features'] > 0:
            feature_info['statistics']['mean_variance'] /= feature_info['total_features']
        
        return feature_info

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features to [0, 1] range using min-max scaling.
        
        Args:
            features: DataFrame containing features to normalize
            
        Returns:
            DataFrame with normalized features
        """
        try:
            normalized_features = features.copy()
            
            for col in normalized_features.columns:
                if normalized_features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    min_val = normalized_features[col].min()
                    max_val = normalized_features[col].max()
                    
                    # Avoid division by zero
                    if max_val != min_val:
                        normalized_features[col] = (normalized_features[col] - min_val) / (max_val - min_val)
                    else:
                        # If all values are the same, set to 0
                        normalized_features[col] = 0.0
            
            return normalized_features
            
        except Exception as e:
            tprint_error(f"❌ Feature normalization failed: {e}")
            return features

    def _standardize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize features to have mean=0 and std=1.
        
        Args:
            features: DataFrame containing features to standardize
            
        Returns:
            DataFrame with standardized features
        """
        try:
            standardized_features = features.copy()
            
            for col in standardized_features.columns:
                if standardized_features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    mean_val = standardized_features[col].mean()
                    std_val = standardized_features[col].std()
                    
                    # Avoid division by zero
                    if std_val > 0:
                        standardized_features[col] = (standardized_features[col] - mean_val) / std_val
                    else:
                        # If std is 0, set to 0
                        standardized_features[col] = 0.0
            
            return standardized_features
            
        except Exception as e:
            tprint_error(f"❌ Feature standardization failed: {e}")
            return features

    def _select_important_features_optimized(self, combined_features: pd.DataFrame, market_data: pd.DataFrame, max_features: int = 100) -> pd.DataFrame:
        """Optimized feature selection using TreeSHAP analysis and mutual information."""

        if combined_features.empty or len(combined_features.columns) <= max_features:
            return combined_features

        try:
            tprint_info(f"🚀 Selecting top {max_features} features from {len(combined_features.columns)} total using TreeSHAP + MI...")
            start_time = time.time()

            # Create target variable (future returns)
            target = market_data['close'].pct_change().shift(-1).fillna(0)
            target = target.loc[combined_features.index]

            # Ensure we have enough data points
            if len(target) < 100:
                tprint_warning("⚠️ Insufficient data for feature selection, using all features")
                return combined_features

            # First, try TreeSHAP analysis if available
            treeshap_features = None
            if TREESHAP_AVAILABLE:
                treeshap_features = self._perform_treeshap_analysis(combined_features, target, max_features)
            
            if treeshap_features is not None and len(treeshap_features) > 0:
                # Use TreeSHAP results
                selected_features = combined_features[treeshap_features]
                elapsed_time = time.time() - start_time
                tprint_success(f"✅ Selected {len(treeshap_features)} features in {elapsed_time:.2f}s using TreeSHAP")
                return selected_features
            else:
                # Fallback to mutual information method
                tprint_warning("⚠️ TreeSHAP analysis failed or unavailable, falling back to mutual information")
                
                # Prepare data for Numba processing
                feature_array = combined_features.fillna(0).values
                target_array = target.values
                
                # Filter out constant features
                feature_std = np.std(feature_array, axis=0)
                valid_features_mask = feature_std > 1e-8
                
                if not np.any(valid_features_mask):
                    tprint_warning("⚠️ No valid features found (all constant)")
                    return combined_features
                
                feature_array_valid = feature_array[:, valid_features_mask]
                valid_column_names = combined_features.columns[valid_features_mask].tolist()
                
                # Calculate mutual information scores using vectorized Numba function
                mi_scores = self._vectorized_mutual_info_batch(feature_array_valid, target_array)
                
                # Create feature-score mapping
                feature_scores = list(zip(valid_column_names, mi_scores))
                
                # Sort by MI score (descending)
                feature_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Select top features
                selected_cols = [col for col, score in feature_scores[:max_features]]
                selected_features = combined_features[selected_cols]

                elapsed_time = time.time() - start_time
                tprint_success(f"✅ Selected {len(selected_cols)} features in {elapsed_time:.2f}s using vectorized MI")
                tprint_info(f"📊 Top 5 features: {[col for col, _ in feature_scores[:5]]}")

                return selected_features

        except Exception as e:
            tprint_error(f"❌ Optimized feature selection failed, falling back to original method: {e}")
            return self._select_important_features(combined_features, market_data, max_features)

    def _select_important_features(self, combined_features: pd.DataFrame, market_data: pd.DataFrame, max_features: int = 100) -> pd.DataFrame:
        """Select most important features before GMM enhancement using mutual information."""

        if combined_features.empty or len(combined_features.columns) <= max_features:
            return combined_features

        try:
            tprint_info(f"🎯 Selecting top {max_features} features from {len(combined_features.columns)} total...")

            # Create target variable (future returns)
            target = market_data['close'].pct_change().shift(-1).fillna(0)
            target = target.loc[combined_features.index]

            # Ensure we have enough data points
            if len(target) < 100:
                tprint_warning("⚠️ Insufficient data for feature selection, using all features")
                return combined_features

            # Calculate mutual information scores
            mi_scores = {}
            valid_columns = []

            for col in combined_features.columns:
                try:
                    # Fill NaN values with 0 for MI calculation
                    feature_data = combined_features[col].fillna(0).values.reshape(-1, 1)

                    # Skip if all values are the same
                    if np.std(feature_data) == 0:
                        continue

                    score = mutual_info_regression(feature_data, target.values, random_state=42)[0]
                    mi_scores[col] = score
                    valid_columns.append(col)

                except Exception as e:
                    tprint_warning(f"⚠️ Could not calculate MI for {col}: {e}")
                    continue

            if not mi_scores:
                tprint_warning("⚠️ No valid features found for selection")
                return combined_features

            # Select top features by MI score
            sorted_features = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
            selected_cols = [col for col, score in sorted_features[:max_features]]

            selected_features = combined_features[selected_cols]

            tprint_success(f"✅ Selected {len(selected_cols)} features with highest mutual information")
            tprint_info(f"📊 Top 5 features: {[col for col, _ in sorted_features[:5]]}")

            return selected_features

        except Exception as e:
            tprint_error(f"❌ Feature selection failed: {e}")
            return combined_features

    # Enhanced GMM integration will be added below
    pass

    async def _save_results(
        self,
        enhanced_features: pd.DataFrame,
        specialist_outputs: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
        ridge_results: Dict[str, Any] = None,
        extratrees_results: Dict[str, Any] = None,
        ensemble_results: Dict[str, Any] = None
    ) -> None:
        """Save enhanced features and specialist outputs."""
         
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = config.get("symbol", "UNKNOWN")
            timeframe = config.get("timeframe", "15m")
            direction = config.get("direction", "long")
             
            # Create outcomes directory
            outcomes_dir = Path("outcomes") / f"specialists_with_gmm_{symbol}_{timeframe}_{direction}_{timestamp}"
            outcomes_dir.mkdir(parents=True, exist_ok=True)
             
            # Skip saving enhanced features CSV to avoid per-timestamp data storage
            # enhanced_features_path = outcomes_dir / f"enhanced_features_{symbol}_{timeframe}.csv"
            # enhanced_features.to_csv(enhanced_features_path, index=True)
            # tprint_success(f"💾 Enhanced features saved to {enhanced_features_path}")
            
            # Save individual specialist outputs
            specialist_dir = outcomes_dir / "specialist_outputs"
            specialist_dir.mkdir(exist_ok=True)
            
            # Save individual specialist outputs to specialist directory
            for specialist_name, outputs in specialist_outputs.items():
                if outputs is not None and (not hasattr(outputs, 'empty') or not outputs.empty):
                    output_path = specialist_dir / f"{specialist_name}_{symbol}_{timeframe}.csv"
                    outputs.to_csv(output_path, index=True)
            
            tprint_success(f"💾 Individual specialist outputs saved to {specialist_dir}")
            
            # Save metadata
            metadata = {
                "timestamp": timestamp,
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction,
                "n_specialists": len(specialist_outputs),
                "enhanced_features_shape": enhanced_features.shape,
                "specialist_names": list(specialist_outputs.keys()),
                "feature_columns": list(enhanced_features.columns)
            }
            
            metadata_path = outcomes_dir / "training_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            tprint_success(f"💾 Metadata saved to {metadata_path}")
            
            # Save Ridge model results
            if ridge_results and ridge_results.get("success", False):
                ridge_dir = outcomes_dir / "ridge_model"
                ridge_dir.mkdir(exist_ok=True)
                
                # Save Ridge model
                import pickle
                ridge_model_path = ridge_dir / "ridge_model.pkl"
                with open(ridge_model_path, 'wb') as f:
                    pickle.dump(ridge_results["model"], f)
                
                # Save Ridge metrics
                ridge_metrics = {
                    "alpha": ridge_results["alpha"],
                    "mse": ridge_results["mse"],
                    "r2_score": ridge_results["r2_score"],
                    "monotonic_features": ridge_results["monotonic_features"],
                    "feature_importance": ridge_results["feature_importance"].to_dict('records')
                }
                
                ridge_metrics_path = ridge_dir / "ridge_metrics.json"
                with open(ridge_metrics_path, 'w') as f:
                    json.dump(ridge_metrics, f, indent=2, default=str)
                
                tprint_success(f"💾 Ridge model saved to {ridge_dir}")
            
            # Save ExtraTrees model results
            if extratrees_results and extratrees_results.get("success", False):
                extratrees_dir = outcomes_dir / "extratrees_model"
                extratrees_dir.mkdir(exist_ok=True)
                
                # Save ExtraTrees model
                extratrees_model_path = extratrees_dir / "extratrees_model.pkl"
                with open(extratrees_model_path, 'wb') as f:
                    pickle.dump(extratrees_results["model"], f)
                
                # Save ExtraTrees metrics
                extratrees_metrics = {
                    "params": extratrees_results["params"],
                    "mse": extratrees_results["mse"],
                    "r2_score": extratrees_results["r2_score"],
                    "feature_importance": extratrees_results["feature_importance"].to_dict('records')
                }
                
                extratrees_metrics_path = extratrees_dir / "extratrees_metrics.json"
                with open(extratrees_metrics_path, 'w') as f:
                    json.dump(extratrees_metrics, f, indent=2, default=str)
                
                tprint_success(f"💾 ExtraTrees model saved to {extratrees_dir}")
            
            # Save Ensemble results (Winner Model)
            if ensemble_results and ensemble_results.get("success", False):
                ensemble_dir = outcomes_dir / "ensemble_winner"
                ensemble_dir.mkdir(exist_ok=True)

                # Save Winner Model
                import pickle
                winner_name = ensemble_results.get("winner_name", "unknown")
                winner_model = ensemble_results.get("winner_model")
                if winner_model:
                    winner_model_path = ensemble_dir / f"{winner_name}_model.pkl"
                    with open(winner_model_path, 'wb') as f:
                        pickle.dump(winner_model, f)

                # Save Ensemble Metrics
                ensemble_metrics = {
                    "winner_name": winner_name,
                    "winner_score": ensemble_results.get("winner_score"),
                    "winner_ic": ensemble_results.get("winner_ic"),
                    "winner_auc": ensemble_results.get("winner_auc"),
                    "all_results": ensemble_results.get("all_results"),
                    "selected_features": ensemble_results.get("selected_features")
                }

                ensemble_metrics_path = ensemble_dir / "ensemble_metrics.json"
                with open(ensemble_metrics_path, 'w') as f:
                    json.dump(ensemble_metrics, f, indent=2, default=str)

                tprint_success(f"💾 Ensemble winner ({winner_name}) saved to {ensemble_dir}")

            # Save composite features to versioned artifacts
            # Note: We use the '_analyst' suffix for the main artifact store
            store_name = f"{symbol}_{config.get('exchange')}_{timeframe}_{direction}_analyst" 
            artifact_store = VersionedArtifactStore(
                Path("versioned_artifacts") / store_name
            )
            
            # Use add_data instead of save
            artifact_store.add_data(
                enhanced_features,
                "specialists_enhanced_features",
                metadata={
                    "n_specialists": len(specialist_outputs),
                    "feature_shape": enhanced_features.shape,
                    "specialist_names": list(specialist_outputs.keys()),
                    "timestamp": datetime.now().isoformat(),
                    "config": config
                }
            )
            
            tprint_success("💾 Enhanced features saved to versioned artifacts")
            
            # --- CRITICAL FIX: Save individual specialist artifacts for diagnostics ---
            # Map specialist step names to expected artifact names
            # Expected by: get_specialist_models_outputs.py and other consumers
            artifact_mapping = {
                # Risk
                "enhanced_ml_risk_regime_step": [
                    "ml_risk_regime_probabilities_15m", 
                    "ml_risk_training_data_15m"
                ],
                # Liquidity
                "enhanced_ml_liquidity_regime_step": [
                    "ml_liquidity_regime_probs_15m" 
                ],
                # Breakout / Bounce (Note: Import uses 'EnhancedMLBreakoutBounceStep' but check usage)
                # Assuming key matches SPECIALIST_IMPORTS key or step name
                "enhanced_ml_breakout_bounce_step": [
                    "ml_breakout_bounce_training_data_15m"
                ],
                # Path
                "enhanced_ml_path_regime_step": [
                    "ml_path_training_data_15m"
                ],
                # Macro
                "enhanced_xgb_macro_regime_step": [
                    "hmm_macro_trend_training_data_15m"
                ],
                # Meso (New - Standardizing name)
                "enhanced_xgb_meso_regime_step": [
                    "xgb_meso_regime_training_data_15m"
                ],
                 # Volatility Burst (New - Standardizing name)
                "enhanced_ml_volatility_burst_step": [
                    "ml_volatility_burst_training_data_15m"
                ],
                 # Volume Force
                "enhanced_ml_volume_force_step": [
                    "ml_volume_force_predictions"
                ],
                 # SMC
                "enhanced_ml_smc_regime_step": [
                    "smc_predictions_with_confidence"
                ],
                 # Spectral (New - Standardizing name)
                "enhanced_ml_spectral_step": [
                     "ml_spectral_training_data_15m"
                ],
                # Microstructure (New - Standardizing name)
                "enhanced_ml_microstructure_step": [
                     "ml_microstructure_training_data_15m"
                ],
                # Momentum Persistence (New - Standardizing name)
                "enhanced_ml_momentum_persistence_step": [
                     "ml_momentum_persistence_training_data_15m"
                ],
                # Candlestick (New)
                "enhanced_ml_candlestick_step": [
                    "ml_candlestick_training_data_15m"
                ],
                # Reversion (New)
                "enhanced_ml_reversion_regime_step": [
                    "ml_reversion_training_data_15m"
                ]
            }
            
            base_timeframe = timeframe  # '15m' typically
            
            tprint_info("🔗 Registering individual specialist artifacts...")
            
            for specialist_name, outputs in specialist_outputs.items():
                if outputs is None or outputs.empty:
                    continue
                    
                # Determine artifact name(s)
                artifact_names = []
                
                # Check specific mapping first
                if specialist_name in artifact_mapping:
                    # Replace '15m' with actual timeframe if different, though usually it is 15m
                    mapped_names = artifact_mapping[specialist_name]
                    # Dynamically replace '15m' with current timeframe if needed, 
                    # but get_specialist_models_outputs hardcodes '_15m' often.
                    # We will use the mapped names as-is because they match get_specialist_models_outputs.
                    artifact_names.extend(mapped_names)
                else:
                    # Default naming convention: {step_name}_outputs
                    artifact_names.append(f"{specialist_name}_outputs")
                
                # Also save standard name consistent with step name for future-proofing
                artifact_names.append(f"{specialist_name}_training_data_{base_timeframe}")
                
                # Attempt to save to appropriate store
                # Most specialists have their own store, or share the analyst store?
                # get_specialist_models_outputs loads from ANY store that contains the artifact name.
                # So saving to the analyst store is fine, OR we can save to specialist-specific stores.
                # Saving to the generic analyst store for now to ensure visibility.
                
                for art_name in set(artifact_names): # Unique names
                    try:
                        artifact_store.save(
                            outputs,
                            art_name,
                            metadata={
                                "source_step": specialist_name,
                                "timestamp": datetime.now().isoformat(),
                                "rows": len(outputs)
                            }
                        )
                        tprint_success(f"  ✅ Saved artifact: {art_name}")
                    except Exception as e:
                        tprint_warning(f"  ⚠️ Failed to save artifact {art_name}: {e}")

        except Exception as e:
            tprint_error(f"❌ Failed to save results: {e}")
    async def _apply_gmm_enhancement(
        self,
        features: pd.DataFrame,
        market_data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """Apply comprehensive GMM enhancement with persistence and adaptive parameters."""
        
        try:
            tprint_info("🧠 Starting GMM Enhancement Pipeline...")
            tprint_info(f"📊 Input features shape: {features.shape}, Market data shape: {market_data.shape}")
            
            from src.training.steps.market_analysis.quantitative_regime_engine import QuantitativeRegimeEngine
            
            # Step 1: Check feature quality
            tprint_info("🔍 Step 1/5: Checking feature quality before GMM enhancement...")
            cleaned_features, quality_report = self._check_feature_quality(features)
            
            tprint_info(f"📊 Feature quality report:")
            tprint_info(f"   - Total features: {quality_report['total_features']}")
            tprint_info(f"   - Constant features removed: {len(quality_report['constant_features'])}")
            tprint_info(f"   - Low variance features: {len(quality_report['low_variance_features'])}")
            tprint_info(f"   - Features kept: {quality_report['final_feature_count']}")
            
            if len(quality_report['constant_features']) > 0:
                tprint_warning(f"⚠️ Removed constant features: {quality_report['constant_features']}")
            
            if len(quality_report['low_variance_features']) > 0:
                tprint_warning(f"⚠️ Found {len(quality_report['low_variance_features'])} features with low variance")
            
            # Step 2: Normalize and standardize features
            tprint_info("🔧 Step 2/5: Normalizing and standardizing features...")
            normalized_features = self._normalize_features(cleaned_features)
            standardized_features = self._standardize_features(normalized_features)
            tprint_success(f"✅ Features standardized: {standardized_features.shape}")
            
            # Use standardized features for GMM
            processed_features = standardized_features
            
            # Generate data hash for caching
            data_hash = self._generate_data_hash(processed_features)
            model_key = f"gmm_{data_hash}"
            tprint_info(f"🔑 Generated data hash: {data_hash[:8]}...")

            # #region agent log - Hypothesis B: GMM caching operations
            with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "id": "log_gmm_cache_start",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "train_specialists_with_gmm_step.py:_apply_gmm_enhancement",
                    "message": "Starting GMM model caching operations",
                    "data": {"data_hash": data_hash, "model_key": model_key, "features_shape": features.shape},
                    "sessionId": "debug-session",
                    "runId": "initial",
                    "hypothesisId": "B"
                }) + '\n')
            # #endregion

            # Try to load cached model
            cached_model, cached_metadata = self._load_gmm_model(model_key)

            # #region agent log - Hypothesis B: Cached model loading result
            with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "id": "log_gmm_cache_loaded",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "train_specialists_with_gmm_step.py:_apply_gmm_enhancement",
                    "message": "Cached model loading result",
                    "data": {
                        "cached_model_found": cached_model is not None,
                        "cached_metadata_keys": list(cached_metadata.keys()) if cached_metadata else None,
                        "model_cache_size": len(self.gmm_model_cache)
                    },
                    "sessionId": "debug-session",
                    "runId": "initial",
                    "hypothesisId": "B"
                }) + '\n')
            # #endregion
            
            # Calculate adaptive parameters
            # #region agent log - Hypothesis C: Adaptive parameter calculation start
            with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "id": "log_adaptive_params_start",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "train_specialists_with_gmm_step.py:_apply_gmm_enhancement",
                    "message": "Starting adaptive parameter calculation",
                    "data": {"market_data_shape": market_data.shape, "features_shape": features.shape},
                    "sessionId": "debug-session",
                    "runId": "initial",
                    "hypothesisId": "C"
                }) + '\n')
            # #endregion

            adaptive_config = self._calculate_adaptive_parameters(market_data, features)

            # #region agent log - Hypothesis C: Adaptive parameter calculation result
            with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "id": "log_adaptive_params_result",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "train_specialists_with_gmm_step.py:_apply_gmm_enhancement",
                    "message": "Adaptive parameter calculation completed",
                    "data": {
                        "adaptive_config_keys": list(adaptive_config.keys()),
                        "n_components": adaptive_config.get("n_components"),
                        "subsample_size": adaptive_config.get("subsample_size"),
                        "wavelet": adaptive_config.get("wavelet")
                    },
                    "sessionId": "debug-session",
                    "runId": "initial",
                    "hypothesisId": "C"
                }) + '\n')
            # #endregion
            
            # Check if we should use cached model or train new one
            tprint_info("🔍 Step 3/5: Checking model cache...")
            use_cached = False
            if cached_model is not None and cached_metadata is not None:
                if self._should_update_model(cached_metadata, len(features)):
                    tprint_info("🔄 Cached model needs update, training new model...")
                else:
                    tprint_info("✅ Using cached GMM model...")
                    use_cached = True
            else:
                tprint_info("🆕 No cached model found, training new model...")
            
            if use_cached:
                # Use cached model for enhancement
                tprint_info("⚡ Using cached model for transformation...")
                engine = cached_model
                results = await engine.transform(market_data, features)
            else:
                # Train new model with adaptive parameters
                tprint_info("🧠 Step 4/5: Training new GMM model...")
                tprint_info(f"🎯 Adaptive config: n_components={adaptive_config.get('n_components')}, subsample={adaptive_config.get('subsample_size')}")
                
                engine = QuantitativeRegimeEngine(**adaptive_config)
                
                tprint_info("🧠 Running comprehensive GMM enhancement with adaptive parameters...")
                results = await engine.fit_transform(market_data, features)
                
                # Save model to cache if successful
                if results.get("success", False):
                    metadata = {
                        "config": adaptive_config,
                        "data_shape": market_data.shape,
                        "features_shape": features.shape,
                        "performance_score": results.get("performance_score", 0.0),
                        "data_hash": data_hash,
                        "specialist_performance": self._specialist_metrics,
                        "ensemble_performance": self._aggregate_specialist_metrics()
                    }
                    self._save_gmm_model(model_key, engine, metadata)
                    
                    # Add to in-memory cache
                    self.gmm_model_cache[model_key] = engine
                    self.gmm_model_metadata[model_key] = metadata
                    
                    # Limit cache size
                    if len(self.gmm_model_cache) > self.gmm_config.max_models_cached:
                        oldest_key = list(self.gmm_model_cache.keys())[0]
                        del self.gmm_model_cache[oldest_key]
                        if oldest_key in self.gmm_model_metadata:
                            del self.gmm_model_metadata[oldest_key]
            
            if results.get("success", False):
                # Step 5: Combine features from both pipelines
                tprint_info("🔗 Step 5/5: Combining GMM pipeline features...")
                combined_features = []
                
                returns_features = results.get("returns_pipeline", {}).get("features")
                fracdiff_features = results.get("fracdiff_pipeline", {}).get("features")
                
                # List GMM features with information
                gmm_features_info = []
                
                if returns_features is not None:
                    returns_info = self._list_gmm_features(returns_features, "returns_pipeline")
                    gmm_features_info.append(returns_info)
                    combined_features.append(returns_features.add_prefix("RETURNS_"))
                    tprint_success(f"✅ Returns pipeline: {len(returns_features.columns)} features")
                    tprint_info(f"   - Mean variance: {returns_info['statistics']['mean_variance']:.6f}")
                    tprint_info(f"   - Min variance: {returns_info['statistics']['min_variance']:.6f}")
                    tprint_info(f"   - Max variance: {returns_info['statistics']['max_variance']:.6f}")
                
                if fracdiff_features is not None:
                    fracdiff_info = self._list_gmm_features(fracdiff_features, "fracdiff_pipeline")
                    gmm_features_info.append(fracdiff_info)
                    combined_features.append(fracdiff_features.add_prefix("FRACDIFF_"))
                    tprint_success(f"✅ FracDiff pipeline: {len(fracdiff_features.columns)} features")
                    tprint_info(f"   - Mean variance: {fracdiff_info['statistics']['mean_variance']:.6f}")
                    tprint_info(f"   - Min variance: {fracdiff_info['statistics']['min_variance']:.6f}")
                    tprint_info(f"   - Max variance: {fracdiff_info['statistics']['max_variance']:.6f}")
                
                # Save GMM features information to results
                if gmm_features_info:
                    results["gmm_features_info"] = gmm_features_info
                
                if combined_features:
                    enhanced = pd.concat(combined_features, axis=1)
                    
                    # Add comparison info
                    comparison = results.get("comparison", {})
                    tprint_info(f"📊 Pipeline comparison: {comparison}")
                    
                    # Memory optimization for enhanced features
                    enhanced = self._optimize_enhanced_features(enhanced)

                    # Generate detailed GMM Report
                    try:
                        from src.training.steps.market_analysis.gmm_report_generator import GMMReportGenerator
                        reporter = GMMReportGenerator()
                        reporter.generate_report(results, symbol=self.context.get("symbol", "UNKNOWN"), timeframe=self.context.get("timeframe", "UNKNOWN"))
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to generate GMM report: {e}")
                    
                    return enhanced
                else:
                    tprint_warning("⚠️ No features generated from GMM enhancement")
                    return features
            else:
                tprint_warning(f"⚠️ GMM enhancement failed: {results.get('error', 'Unknown error')}")
                return features
                
        except Exception as e:
            tprint_error(f"❌ Comprehensive GMM enhancement failed: {e}")
            return features

    def _generate_enhanced_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced features that all specialists need."""
        try:
            from src.training.steps.market_analysis.enhanced_feature_generators import NonLinearFeatureGenerator

            # Start with market data, but exclude non-numeric columns that could cause issues
            enhanced_data = market_data.copy()

            # Identify and separate non-numeric columns that should not be processed
            non_numeric_cols = {}
            cols_to_drop = []

            for col in enhanced_data.columns:
                if enhanced_data[col].dtype == 'object':
                    # Keep string columns that are metadata
                    if col.lower() in ['symbol', 'exchange', 'interval', 'day']:
                        non_numeric_cols[col] = enhanced_data[col].copy()
                        cols_to_drop.append(col)
                    else:
                        # Drop other object columns that might cause issues
                        cols_to_drop.append(col)
                elif enhanced_data[col].dtype.name == 'category':
                    # Convert categorical to string and store
                    non_numeric_cols[col] = enhanced_data[col].astype(str).copy()
                    cols_to_drop.append(col)

            # Remove non-numeric columns temporarily for feature generation
            if cols_to_drop:
                enhanced_data = enhanced_data.drop(columns=cols_to_drop)

            # Generate polynomial features for key columns
            nonlinear_gen = NonLinearFeatureGenerator()

            # Key columns that specialists expect polynomial features for
            key_columns = ['high', 'low', 'open', 'close', 'volume']
            available_columns = [col for col in key_columns if col in enhanced_data.columns and
                               enhanced_data[col].dtype in ['float64', 'float32', 'int64', 'int32']]

            if available_columns:
                tprint_info(f"🔧 Generating polynomial features for: {available_columns}")
                poly_features = nonlinear_gen.add_polynomial_features(enhanced_data, available_columns, degree=3)
                if not poly_features.empty:
                    # Add polynomial features
                    enhanced_data = pd.concat([enhanced_data, poly_features], axis=1)
                    tprint_info(f"✅ Added {len(poly_features.columns)} polynomial features")

            # Generate additional features that specialists commonly need
            numeric_cols = enhanced_data.select_dtypes(include=[np.number]).columns

            # Add some basic derived features that specialists might expect
            if 'high' in numeric_cols and 'low' in numeric_cols:
                enhanced_data['range'] = enhanced_data['high'] - enhanced_data['low']
                enhanced_data['range_squared'] = enhanced_data['range'] ** 2
                enhanced_data['range_cubed'] = enhanced_data['range'] ** 3

            if 'close' in numeric_cols and 'open' in numeric_cols:
                enhanced_data['body'] = abs(enhanced_data['close'] - enhanced_data['open'])
                enhanced_data['body_squared'] = enhanced_data['body'] ** 2
                enhanced_data['body_cubed'] = enhanced_data['body'] ** 3

            # Add volume-based features if volume exists
            if 'volume' in numeric_cols:
                enhanced_data['volume_sqrt'] = np.sqrt(enhanced_data['volume'] + 1e-8)
                enhanced_data['volume_log'] = np.log1p(enhanced_data['volume'])

            # Clean up any NaN/inf values
            enhanced_data = enhanced_data.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Restore non-numeric columns
            for col, data in non_numeric_cols.items():
                enhanced_data[col] = data

            tprint_success(f"✅ Generated {len(enhanced_data.columns)} total features (original: {len(market_data.columns)})")

            return enhanced_data

        except Exception as e:
            tprint_error(f"❌ Enhanced feature generation failed: {e}")
            import traceback
            tprint_error(f"Traceback: {traceback.format_exc()}")
            # Return original data if enhancement fails
            return market_data

    def _optimize_enhanced_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Optimize enhanced features for memory efficiency."""
        try:
            # Downcast numeric columns
            for col in features.columns:
                if features[col].dtype == 'float64':
                    features[col] = pd.to_numeric(features[col], downcast='float')
                elif features[col].dtype == 'int64':
                    features[col] = pd.to_numeric(features[col], downcast='integer')

            # Identify and convert sparse features
            sparse_cols = []
            for col in features.columns:
                if features[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                    sparsity = (features[col] == 0).mean()
                    if sparsity > self.memory_config.sparse_threshold:
                        sparse_cols.append(col)

            if sparse_cols:
                tprint_info(f"🎯 Converting {len(sparse_cols)} sparse features to optimized format")
                # Note: Could use pandas.SparseDtype here if needed
                # For now, just identify them for potential future optimization

            return features

        except Exception as e:
            tprint_warning(f"⚠️ Feature optimization failed: {e}")
            return features

    def _aggregate_specialist_metrics(self) -> Dict[str, Any]:
        """Aggregate performance metrics across all trained specialists."""
        if not self._specialist_metrics:
            return {}

        try:
            # Extract metric values
            auc_scores = [m.get('auc', 0) for m in self._specialist_metrics.values() if m.get('auc', 0) > 0]
            mi_scores = [m.get('mi_score', 0) for m in self._specialist_metrics.values() if m.get('mi_score', 0) > 0]
            best_auc_scores = [m.get('best_auc', 0) for m in self._specialist_metrics.values() if m.get('best_auc', 0) > 0]
            best_mi_scores = [m.get('best_mi', 0) for m in self._specialist_metrics.values() if m.get('best_mi', 0) > 0]

            # Calculate aggregate statistics
            ensemble_metrics = {
                'total_specialists': len(self._specialist_metrics),
                'specialists_with_auc': len(auc_scores),
                'specialists_with_mi': len(mi_scores),
            }

            if auc_scores:
                ensemble_metrics.update({
                    'mean_auc': float(np.mean(auc_scores)),
                    'std_auc': float(np.std(auc_scores)),
                    'best_auc_overall': float(max(auc_scores)),
                    'worst_auc': float(min(auc_scores)),
                    'auc_above_55': sum(1 for auc in auc_scores if auc > 0.55),
                    'auc_above_60': sum(1 for auc in auc_scores if auc > 0.60),
                })

            if mi_scores:
                ensemble_metrics.update({
                    'mean_mi': float(np.mean(mi_scores)),
                    'std_mi': float(np.std(mi_scores)),
                    'best_mi_overall': float(max(mi_scores)),
                    'worst_mi': float(min(mi_scores)),
                    'mi_above_01': sum(1 for mi in mi_scores if mi > 0.01),
                    'mi_above_02': sum(1 for mi in mi_scores if mi > 0.02),
                })

            if best_auc_scores:
                ensemble_metrics['best_auc_mean'] = float(np.mean(best_auc_scores))

            if best_mi_scores:
                ensemble_metrics['best_mi_mean'] = float(np.mean(best_mi_scores))

            # Compliance and readiness metrics
            ensemble_metrics.update({
                'compliant_specialists': sum(1 for m in self._specialist_metrics.values() if m.get('compliance', False)),
                'ensemble_ready_specialists': sum(1 for m in self._specialist_metrics.values() if m.get('ensemble_ready', False)),
                'training_success_rate': sum(1 for m in self._specialist_metrics.values() if m.get('training_success', False)) / len(self._specialist_metrics),
            })

            return ensemble_metrics

        except Exception as e:
            tprint_warning(f"⚠️ Metrics aggregation failed: {e}")
            return {
                'total_specialists': len(self._specialist_metrics),
                'aggregation_error': str(e)
            }

    def _save_specialist_metrics_csv(self, symbol: str, exchange: str, timeframe: str, direction: str) -> None:
        """Save comprehensive specialist metrics to CSV file with datetime stamp."""
        try:
            import pandas as pd
            from datetime import datetime

            if not self._specialist_metrics:
                tprint_warning("⚠️ No specialist metrics to save")
                return

            # Prepare data for CSV
            csv_data = []

            for specialist_name, metrics in self._specialist_metrics.items():
                row = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'direction': direction,
                    'specialist_name': specialist_name,
                    'training_success': metrics.get('training_success', False),
                    'auc': metrics.get('auc', 0.0),
                    'mi_score': metrics.get('mi_score', 0.0),
                    'best_auc': metrics.get('best_auc', 0.0),
                    'best_mi': metrics.get('best_mi', 0.0),
                    'n_features': metrics.get('n_features', 0),
                    'n_samples': metrics.get('n_samples', 0),
                    'compliance': metrics.get('compliance', False),
                    'enhanced_mi_score': metrics.get('enhanced_mi_score', 0.0),
                    'ensemble_ready': metrics.get('ensemble_ready', False),
                    'hyperparams_n_estimators': metrics.get('hyperparams', {}).get('n_estimators', 0),
                    'hyperparams_max_depth': metrics.get('hyperparams', {}).get('max_depth', 0),
                    'hyperparams_learning_rate': metrics.get('hyperparams', {}).get('learning_rate', 0.0),
                    'hyperparams_subsample': metrics.get('hyperparams', {}).get('subsample', 0.0),
                    'hyperparams_colsample_bytree': metrics.get('hyperparams', {}).get('colsample_bytree', 0.0),
                    'hyperparams_gamma': metrics.get('hyperparams', {}).get('gamma', 0.0),
                    'hyperparams_reg_alpha': metrics.get('hyperparams', {}).get('reg_alpha', 0.0),
                    'hyperparams_reg_lambda': metrics.get('hyperparams', {}).get('reg_lambda', 0.0),
                    'hyperparams_min_child_weight': metrics.get('hyperparams', {}).get('min_child_weight', 0),
                }
                csv_data.append(row)

            # Create DataFrame and save to CSV
            df = pd.DataFrame(csv_data)

            # Generate filename with datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"specialist_metrics_{symbol}_{timestamp}.csv"
            filepath = Path("outcomes") / filename

            # Ensure outcomes directory exists
            Path("outcomes").mkdir(exist_ok=True)

            # Save CSV
            df.to_csv(filepath, index=False)

            tprint_success(f"✅ Specialist metrics saved to {filepath}")
            tprint_info(f"📊 Saved metrics for {len(csv_data)} specialists with {len(df.columns)} metrics each")

        except Exception as e:
            tprint_error(f"❌ Failed to save specialist metrics CSV: {e}")
            import traceback
            tprint_error(f"Traceback: {traceback.format_exc()}")
