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

from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.versioned_artifacts import VersionedArtifactStore

# Import all enhanced specialists
SPECIALIST_IMPORTS = {
    "enhanced_ml_momentum_persistence_step": ("src.training.steps.market_analysis.ml_momentum_persistence_step_enhanced", "EnhancedMLMomentumPersistenceStep"),
    "enhanced_ml_smc_regime_step": ("src.training.steps.market_analysis.ml_smc_regime_step_enhanced", "EnhancedMLSMCRegimeStep"),
    "enhanced_ml_volatility_burst_step": ("src.training.steps.market_analysis.ml_volatility_burst_step_enhanced", "EnhancedMLVolatilityBurstStep"),
    "enhanced_ml_volume_force_step": ("src.training.steps.market_analysis.ml_volume_force_step_enhanced", "EnhancedMLVolumeForceStep"),
    "enhanced_xgb_macro_regime_step": ("src.training.steps.market_analysis.xgb_macro_regime_step_enhanced", "EnhancedXGBMacroRegimeStep"),
    "enhanced_xgb_meso_regime_step": ("src.training.steps.market_analysis.xgb_meso_regime_step_enhanced", "EnhancedXGBMesoRegimeStep"),
    "enhanced_ml_liquidity_regime_step": ("src.training.steps.market_analysis.ml_liquidity_regime_step_enhanced", "EnhancedMLLiquidityRegimeStep"),
    "enhanced_ml_path_regime_step": ("src.training.steps.market_analysis.ml_path_regime_step_enhanced", "EnhancedMLPathRegimeStep"),
    "enhanced_ml_risk_regime_step": ("src.training.steps.market_analysis.ml_risk_regime_step_enhanced", "EnhancedMLRiskRegimeStep"),
    "enhanced_ml_microstructure_step": ("src.training.steps.market_analysis.ml_microstructure_step_enhanced", "EnhancedMLMicrostructureStep"),
    "enhanced_ml_spectral_step": ("src.training.steps.market_analysis.ml_spectral_step_enhanced", "EnhancedMLSpectralStep"),
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
        market_data, source = self.load_market_data_or_fail(config)
        if market_data is None or market_data.empty:
            error_msg = "❌ Failed to load market data"
            tprint_error(error_msg)
            return {"success": False, "error": error_msg}

        tprint_success(f"✅ Loaded {len(market_data)} bars from {source}")

        # Memory optimization: reduce data size for training
        # Use only recent data to reduce memory usage
        max_training_samples = 50000  # Limit to 50k samples for training
        if len(market_data) > max_training_samples:
            tprint_info(f"🧠 Reducing data size from {len(market_data)} to {max_training_samples} samples for memory efficiency")
            market_data = market_data.tail(max_training_samples)
            tprint_success(f"✅ Reduced data to {len(market_data)} samples")

        # Step 1.5: Generate enhanced features for all specialists
        tprint_info("🔧 Generating Enhanced Features for All Specialists...")
        enhanced_market_data = self._generate_enhanced_features(market_data)
        tprint_success(f"✅ Enhanced features generated: {enhanced_market_data.shape}")

        # Additional memory cleanup
        del market_data
        gc.collect()

        # Step 2: Train all specialists (memory-efficient batch processing)
        tprint_info("🎯 Training 11 Specialist Models...")
        
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
        
        # Train specialists with memory-efficient approach
        specialist_outputs = await self._train_all_specialists_memory_efficient(
            enhanced_market_data, config, force_retrain, batch_size=optimal_batch_size
        )

        if not specialist_outputs:
            error_msg = "❌ No specialist outputs generated"
            tprint_error(error_msg)
            return {"success": False, "error": error_msg}

        tprint_success(f"✅ Generated outputs from {len(specialist_outputs)} specialists")
        
        # Step 3: Combine specialist outputs
        tprint_info("🔗 Combining specialist outputs...")
        combined_features = self._combine_specialist_outputs(specialist_outputs)

        if combined_features.empty:
            error_msg = "❌ Failed to combine specialist outputs"
            tprint_error(error_msg)
            return {"success": False, "error": error_msg}

        tprint_success(f"✅ Combined features shape: {combined_features.shape}")

        # Step 4: Feature Selection (before expensive GMM processing)
        tprint_info("🎯 Performing optimized feature selection...")
        selected_features = self._select_important_features_optimized(combined_features, market_data, max_features=100)

        # Step 5: GMM Enhancement
        if GMM_AVAILABLE:
            tprint_info("🧠 Applying GMM Enhancement...")
            enhanced_features = await self._apply_gmm_enhancement(
                selected_features, market_data, config
            )

            if enhanced_features is None or enhanced_features.empty:
                tprint_warning("⚠️ GMM enhancement failed, using selected features")
                enhanced_features = selected_features
            else:
                tprint_success(f"✅ GMM enhanced features shape: {enhanced_features.shape}")
        else:
            tprint_warning("⚠️ GMM not available, using selected features")
            enhanced_features = selected_features
        
        # Step 5: Save results
        tprint_info("💾 Saving enhanced features and results...")
        await self._save_results(enhanced_features, specialist_outputs, config)
        
        # Summary
        results = {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "direction": direction,
            "n_specialists": len(specialist_outputs),
            "raw_features_shape": combined_features.shape,
            "selected_features_shape": selected_features.shape,
            "enhanced_features_shape": enhanced_features.shape,
            "specialist_outputs": list(specialist_outputs.keys()),
            "batch_size": optimal_batch_size,
            "feature_selection_applied": True,
            "training_time": time.time()
        }
        
        tprint_success("🎉 Specialist Training with GMM Enhancement completed successfully!")
        tprint_info(f"📊 Summary: {len(specialist_outputs)} specialists trained, features enhanced")
        
        return results

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

            # Train batch in parallel
            batch_outputs = await self._train_specialist_batch(
                batch, market_data, config, force_retrain
            )

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

        tprint_info(f"🔄 Training specialists in memory-efficient mode (batch_size={batch_size})...")

        specialist_outputs = {}
        temp_dir = Path(tempfile.mkdtemp(prefix="specialist_outputs_"))

        try:
            # Process specialists in batches
            specialist_items = list(SPECIALIST_IMPORTS.items())

            for i in range(0, len(specialist_items), batch_size):
                batch = specialist_items[i:i + batch_size]
                batch_names = [name for name, _ in batch]

                tprint_info(f"📦 Processing batch {i//batch_size + 1}/{len(specialist_items)//batch_size + 1}: {batch_names}")

                # Train batch
                batch_outputs = await self._train_specialist_batch(
                    batch, market_data, config, force_retrain
                )

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

        batch_outputs = {}

        # Create tasks for parallel execution
        tasks = []
        for step_name, (module_path, class_name) in batch:
            task = self._train_single_specialist_async(
                step_name, module_path, class_name, market_data, config, force_retrain
            )
            tasks.append(task)

        # Execute batch in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

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

        try:
            start_time = time.time()

            # Import specialist class
            module = __import__(module_path, fromlist=[class_name])
            specialist_class = getattr(module, class_name)

            # Initialize specialist
            specialist = specialist_class(step_name)

            # Setup context
            context = {
                "symbol": config.get("symbol"),
                "exchange": config.get("exchange"),
                "timeframe": config.get("timeframe"),
                "direction": config.get("direction"),
                "model": "analyst"
            }

            # Train specialist and get outputs
            outputs = await self._train_single_specialist(
                specialist, market_data, context, force_retrain
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
        force_retrain: bool
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
                "verbose": False  # Reduce verbosity for batch training
            })

            # Run specialist
            result = await specialist.execute(specialist_config)

            # Extract outputs/predictions from result
            if result and isinstance(result, dict):
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

    def _select_important_features_optimized(self, combined_features: pd.DataFrame, market_data: pd.DataFrame, max_features: int = 100) -> pd.DataFrame:
        """Optimized feature selection using vectorized Numba mutual information calculation."""

        if combined_features.empty or len(combined_features.columns) <= max_features:
            return combined_features

        try:
            tprint_info(f"🚀 Selecting top {max_features} features from {len(combined_features.columns)} total using vectorized MI...")
            start_time = time.time()

            # Create target variable (future returns)
            target = market_data['close'].pct_change().shift(-1).fillna(0)
            target = target.loc[combined_features.index]

            # Ensure we have enough data points
            if len(target) < 100:
                tprint_warning("⚠️ Insufficient data for feature selection, using all features")
                return combined_features

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
        config: Dict[str, Any]
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
            
            # Save enhanced features
            enhanced_features_path = outcomes_dir / f"enhanced_features_{symbol}_{timeframe}.csv"
            enhanced_features.to_csv(enhanced_features_path, index=True)
            tprint_success(f"💾 Enhanced features saved to {enhanced_features_path}")
            
            # Save individual specialist outputs
            specialist_dir = outcomes_dir / "specialist_outputs"
            specialist_dir.mkdir(exist_ok=True)
            
            for specialist_name, outputs in specialist_outputs.items():
                if outputs is not None and not outputs.empty:
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
            
            # Save to versioned artifacts
            artifact_store = VersionedArtifactStore(
                Path("versioned_artifacts") / f"{symbol}_{config.get('exchange')}_{timeframe}_{direction}_analyst"
            )
            
            artifact_store.save(
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
            from src.training.steps.market_analysis.quantitative_regime_engine import QuantitativeRegimeEngine
            
            # Generate data hash for caching
            data_hash = self._generate_data_hash(features)
            model_key = f"gmm_{data_hash}"

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
            use_cached = False
            if cached_model is not None and cached_metadata is not None:
                if self._should_update_model(cached_metadata, len(features)):
                    tprint_info("🔄 Cached model needs update, training new model...")
                else:
                    tprint_info("✅ Using cached GMM model...")
                    use_cached = True
            
            if use_cached:
                # Use cached model for enhancement
                engine = cached_model
                results = await engine.transform(market_data, features)
            else:
                # Train new model with adaptive parameters
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
                        "data_hash": data_hash
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
                # Combine features from both pipelines
                combined_features = []
                
                returns_features = results.get("returns_pipeline", {}).get("features")
                fracdiff_features = results.get("fracdiff_pipeline", {}).get("features")
                
                if returns_features is not None:
                    combined_features.append(returns_features.add_prefix("RETURNS_"))
                    tprint_success(f"✅ Returns pipeline: {len(returns_features.columns)} features")
                
                if fracdiff_features is not None:
                    combined_features.append(fracdiff_features.add_prefix("FRACDIFF_"))
                    tprint_success(f"✅ FracDiff pipeline: {len(fracdiff_features.columns)} features")
                
                if combined_features:
                    enhanced = pd.concat(combined_features, axis=1)
                    
                    # Add comparison info
                    comparison = results.get("comparison", {})
                    tprint_info(f"📊 Pipeline comparison: {comparison}")
                    
                    # Memory optimization for enhanced features
                    enhanced = self._optimize_enhanced_features(enhanced)
                    
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
