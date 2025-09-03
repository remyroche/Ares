"""Step 6: Advanced Feature Engineering - Refactored and modular.

This module generates advanced features including technical indicators,
wavelet features, and market microstructure features.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.decorators import handles_errors
from src.training.base_step import BaseStep

# Import feature engineering utilities
from src.training.utils.feature_engineering.resampling import OptimizedResampler
from src.training.utils.feature_engineering.technical_indicators import TechnicalIndicatorCalculator
from src.training.utils.feature_engineering.wavelet_features import WaveletTransformAnalyzer
from src.utils.logger import system_logger


class AdvancedFeatureEngineeringStep(BaseStep):
    """Step 6: Advanced Feature Engineering using modular components."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize advanced feature engineering step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "06", "advanced_feature_engineering")
        
        # Feature engineering components
        self.resampler = None
        self.wavelet_analyzer = None
        self.indicator_calculator = None
        
        # Configuration
        self.feature_config = config.get("feature_engineering", {})
        self.enable_wavelets = self.feature_config.get("enable_wavelets", True)
        self.enable_multi_timeframe = self.feature_config.get("enable_multi_timeframe", True)
        self.timeframes = self.feature_config.get("timeframes", ["5m", "15m", "1h"])
        
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        self.resampler = OptimizedResampler()
        self.wavelet_analyzer = WaveletTransformAnalyzer(cache_enabled=True)
        self.indicator_calculator = TechnicalIndicatorCalculator()
        
        self.logger.info("✅ Feature engineering components initialized")
    
    def validate_inputs(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, list]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for labeled data from previous step
        if "labeled_data" not in pipeline_state:
            errors.append("No labeled_data found in pipeline state")
        
        # Check for required columns
        if "labeled_data" in pipeline_state:
            data_path = Path(pipeline_state["labeled_data"])
            if data_path.exists():
                try:
                    # Load a small sample to check columns
                    sample = pd.read_parquet(data_path, columns=['open', 'high', 'low', 'close', 'volume'])
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing = set(required_cols) - set(sample.columns)
                    if missing:
                        errors.append(f"Missing required columns: {missing}")
                except Exception as e:
                    errors.append(f"Failed to validate data file: {e}")
            else:
                errors.append(f"Labeled data file not found: {data_path}")
        
        return len(errors) == 0, errors
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="advanced feature engineering"
    )
    async def execute_logic(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute advanced feature engineering.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        # Extract parameters
        symbol = training_input["symbol"]
        exchange = training_input["exchange"]
        base_timeframe = training_input.get("timeframe", "1m")
        
        self.logger.info(f"🔧 Engineering advanced features for {symbol} ({base_timeframe})")
        
        # Load labeled data
        labeled_data_path = Path(pipeline_state["labeled_data"])
        data = pd.read_parquet(labeled_data_path)
        
        # Memory optimization: cast to float32 where appropriate
        for c in ["open", "high", "low", "close", "volume"]:
            if c in data.columns and data[c].dtype == np.float64:
                data[c] = data[c].astype(np.float32)
        if "composite_cluster_id" in data.columns:
            data["composite_cluster_id"] = data["composite_cluster_id"].astype("category")
        
        self.logger.info(f"📊 Loaded {len(data)} rows of labeled data")
        
        # Initialize feature storage
        all_features = {}
        
        # 1. Calculate technical indicators (chunked to bound memory)
        self.logger.info("📈 Calculating technical indicators...")
        chunk_size = int(self.feature_config.get("chunk_size", 300_000))
        tech_chunks: List[pd.DataFrame] = []
        for start in range(0, len(data), chunk_size):
            end = min(len(data), start + chunk_size)
            part = data.iloc[start:end].copy()
            tech = self.indicator_calculator.calculate_all_features(part)
            tech_chunks.append(tech)
        technical_features = pd.concat(tech_chunks, axis=0, ignore_index=True)
        all_features['technical'] = technical_features
        
        # 2. Calculate wavelet features (if enabled)
        if self.enable_wavelets:
            self.logger.info("🌊 Calculating wavelet features...")
            wavelet_features = self.wavelet_analyzer.extract_wavelet_features(
                data,
                price_column='close',
                symbol=symbol,
                timeframe=base_timeframe
            )
            all_features['wavelet'] = wavelet_features
        
        # 3. Calculate multi-timeframe features (if enabled)
        if self.enable_multi_timeframe:
            self.logger.info("⏰ Calculating multi-timeframe features...")
            mtf_features = await self._calculate_multi_timeframe_features(
                data, base_timeframe, symbol
            )
            all_features['multi_timeframe'] = mtf_features
        
        # 4. Calculate market microstructure features
        self.logger.info("🔬 Calculating market microstructure features...")
        microstructure_features = self._calculate_microstructure_features(data)
        all_features['microstructure'] = microstructure_features
        
        # 5. Combine all features
        self.logger.info("🔗 Combining all features...")
        combined_features = self._combine_features(data, all_features)
        
        # 6. Handle missing values
        combined_features = self._handle_missing_values(combined_features)
        
        # 7. Split into train/validation sets
        train_features, val_features = self._split_features(
            combined_features,
            pipeline_state.get("train_end_idx", int(len(combined_features) * 0.8))
        )
        
        # 8. Save feature sets with compression
        output_dir = Path(training_input.get("data_dir", "data/training"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / f"{exchange}_{symbol}_{base_timeframe}_features_train.parquet"
        val_path = output_dir / f"{exchange}_{symbol}_{base_timeframe}_features_val.parquet"
        
        train_features.to_parquet(train_path, compression="snappy")
        val_features.to_parquet(val_path, compression="snappy")
        
        self.logger.info(
            f"✅ Saved features - Train: {len(train_features)} rows, "
            f"Val: {len(val_features)} rows, "
            f"Features: {len(train_features.columns)} columns"
        )
        
        # Update pipeline state
        pipeline_state["advanced_features"] = {
            "train": str(train_path),
            "val": str(val_path),
            "n_features": len(train_features.columns),
            "feature_groups": list(all_features.keys()),
            "feature_names": list(train_features.columns)
        }
        
        # Add feature statistics
        pipeline_state["feature_statistics"] = self._calculate_feature_statistics(
            train_features
        )
        
        return pipeline_state
    
    async def _calculate_multi_timeframe_features(
        self,
        data: pd.DataFrame,
        base_timeframe: str,
        symbol: str
    ) -> pd.DataFrame:
        """Calculate features from multiple timeframes."""
        mtf_features = pd.DataFrame(index=data.index)
        
        # Get multi-timeframe data
        mtf_data = self.resampler.create_multi_timeframe_features(
            data, base_timeframe, self.timeframes
        )
        
        # Calculate indicators for each timeframe
        for tf, tf_data in mtf_data.items():
            if tf == base_timeframe:
                continue
                
            # Calculate basic features for this timeframe
            tf_indicators = self.indicator_calculator.calculate_all_features(tf_data)
            
            # Select key features to avoid explosion
            key_features = [
                'returns', 'volatility_20', 'rsi_14', 'macd_hist',
                'volume_ratio_10', 'bb_position_20'
            ]
            
            # Align to base timeframe
            for feat in key_features:
                if feat in tf_indicators.columns:
                    # Resample back to base timeframe
                    aligned = tf_indicators[feat].reindex(data.index, method='ffill')
                    mtf_features[f'{feat}_{tf}'] = aligned
        
        return mtf_features
    
    def _calculate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features."""
        features = pd.DataFrame(index=data.index)
        
        # Spread metrics
        features['spread'] = data['high'] - data['low']
        features['spread_pct'] = features['spread'] / data['close']
        features['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
        
        # Volume-weighted metrics
        features['vwap'] = (data['typical_price'] * data['volume']).cumsum() / data['volume'].cumsum()
        features['price_to_vwap'] = data['close'] / features['vwap']
        
        # Liquidity metrics
        features['dollar_volume'] = data['close'] * data['volume']
        features['log_dollar_volume'] = np.log1p(features['dollar_volume'])
        
        # Price impact metrics
        features['price_impact'] = data['close'].pct_change().abs() / (data['volume'] + 1)
        features['kyle_lambda'] = features['price_impact'].rolling(20).mean()
        
        # Order flow imbalance
        features['order_flow_imbalance'] = np.where(
            data['close'] > data['open'],
            data['volume'],
            -data['volume']
        )
        features['ofi_cumsum'] = features['order_flow_imbalance'].cumsum()
        
        return features
    
    def _combine_features(
        self,
        original_data: pd.DataFrame,
        feature_groups: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Combine all feature groups into a single DataFrame."""
        # Start with original OHLCV data
        combined = original_data[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Add labels if present
        if 'label' in original_data.columns:
            combined['label'] = original_data['label']
        
        # Add each feature group
        for group_name, features in feature_groups.items():
            self.logger.info(f"Adding {len(features.columns)} {group_name} features")
            
            # Ensure index alignment
            features = features.reindex(combined.index)
            
            # Add prefix to avoid naming conflicts
            if group_name != 'technical':  # Technical indicators keep original names
                features = features.add_prefix(f'{group_name}_')
            
            # Concatenate
            combined = pd.concat([combined, features], axis=1)
        
        return combined
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Forward fill first (for time series continuity)
        features = features.fillna(method='ffill')
        
        # Then backward fill for any remaining at the start
        features = features.fillna(method='bfill')
        
        # Fill any remaining with 0
        features = features.fillna(0)
        
        # Replace infinities
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def _split_features(
        self,
        features: pd.DataFrame,
        train_end_idx: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split features into train and validation sets."""
        train_features = features.iloc[:train_end_idx]
        val_features = features.iloc[train_end_idx:]
        
        return train_features, val_features
    
    def _calculate_feature_statistics(
        self,
        features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate statistics about the features."""
        numeric_features = features.select_dtypes(include=[np.number])
        
        return {
            "n_samples": len(features),
            "n_features": len(numeric_features.columns),
            "missing_values": numeric_features.isnull().sum().to_dict(),
            "feature_means": numeric_features.mean().to_dict(),
            "feature_stds": numeric_features.std().to_dict(),
            "correlation_matrix_sample": numeric_features.corr().iloc[:5, :5].to_dict()
        }
    
    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for advanced features
        if "advanced_features" not in pipeline_state:
            errors.append("No advanced_features in pipeline state")
            return False, errors
        
        features_info = pipeline_state["advanced_features"]
        
        # Check train and val paths
        for split in ["train", "val"]:
            if split not in features_info:
                errors.append(f"No {split} features path")
            else:
                path = Path(features_info[split])
                if not path.exists():
                    errors.append(f"{split} features file not found: {path}")
        
        # Check feature count
        if features_info.get("n_features", 0) < 10:
            errors.append(f"Too few features: {features_info.get('n_features', 0)}")
        
        return len(errors) == 0, errors
    
    def get_required_inputs(self) -> list:
        """Get list of required inputs for this step."""
        return ["symbol", "exchange", "labeled_data"]
    
    def get_produced_outputs(self) -> list:
        """Get list of outputs produced by this step."""
        return ["advanced_features", "feature_statistics"]
    
    def get_dependencies(self) -> list:
        """Get list of step dependencies."""
        return ["05"]  # Depends on labeling step