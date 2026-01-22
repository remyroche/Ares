"""Global Multi-Asset Meta-Labeling HPO Sample Weighted Step.

This step orchestrates multi-asset training by:
1. Loading data for all specified assets
2. Adding asset-specific features (asset ID, volatility normalization)
3. Combining data into unified training set
4. Running meta-labeling HPO on combined dataset
5. Storing unified model with asset-specific components

Key Features:
- Multi-asset data loading and combination
- Per-asset volatility normalization
- Asset-specific identification features
- Unified model training with asset context
- Asset-specific model components for inference
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from dataclasses import asdict

from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint_success, tprint_warning, tprint_info, tprint_error

from src.training.steps.labeling.meta_labeling_hpo_sample_weighted import (
    MetaLabelingHPOSampleWeightedStep
)


class GlobalMetaLabelingHPOSampleWeightedStep(BaseStep):
    """
    Global multi-asset meta-labeling HPO orchestration step.
    
    This step extends the single-asset MetaLabelingHPOSampleWeightedStep
    to handle multiple assets with asset-specific features and normalization.
    """

    def __init__(self, step_name: str):
        """Initialize the global meta-labeling step."""
        super().__init__(step_name)
        self.asset_data = {}
        self.combined_data = None
        self.asset_stats = {}

    def _load_asset_data(self, config: Dict[str, Any], asset: str) -> pd.DataFrame:
        """Load market data for a specific asset."""
        asset_config = config.copy()
        asset_config['symbol'] = f"{asset}USDT"
        
        tprint_info(f"Loading data for {asset}USDT...")
        
        # Use BaseStep's standard data loading
        market_data, _source = self.load_market_data_or_fail(
            asset_config,
            pipeline_state={},
            allow_config_override=True,
            light_mode_filter=True,
        )
        
        if market_data is None or market_data.empty:
            raise ValueError(f"Failed to load market data for {asset}USDT")
        
        tprint_success(f"✅ Loaded {len(market_data)} rows for {asset}USDT from {_source}")
        return market_data

    def _add_asset_features(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Add asset-specific features to the dataframe."""
        df = df.copy()
        
        # Add asset identifier
        df['asset_id'] = asset
        
        # Add one-hot encoded asset features
        all_assets = self.asset_stats.keys()
        for other_asset in all_assets:
            df[f'asset_{other_asset}'] = (df['asset_id'] == other_asset).astype(int)
        
        return df

    def _normalize_volatility_per_asset(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Apply per-asset volatility normalization."""
        df = df.copy()
        
        # Calculate per-asset volatility statistics
        close_series = pd.to_numeric(df['close'], errors='coerce')
        returns = close_series.pct_change().replace([np.inf, -np.inf], np.nan)
        
        # Rolling volatility (20-period default)
        vol_window = 20
        rolling_vol = returns.rolling(window=vol_window, min_periods=1).std()
        
        # Store asset statistics
        self.asset_stats[asset] = {
            'vol_mean': rolling_vol.mean(),
            'vol_std': rolling_vol.std(),
            'vol_median': rolling_vol.median(),
            'returns_mean': returns.mean(),
            'returns_std': returns.std(),
        }
        
        # Apply volatility normalization
        if self.asset_stats[asset]['vol_std'] > 0:
            vol_normalized = (rolling_vol - self.asset_stats[asset]['vol_mean']) / self.asset_stats[asset]['vol_std']
            df['volatility_normalized'] = vol_normalized.fillna(0)
        else:
            df['volatility_normalized'] = 0
        
        # Add asset-specific volatility regime
        vol_threshold_high = rolling_vol.quantile(0.67)
        vol_threshold_low = rolling_vol.quantile(0.33)
        df['vol_regime_asset'] = np.where(
            rolling_vol > vol_threshold_high, 'high',
            np.where(rolling_vol < vol_threshold_low, 'low', 'medium')
        )
        
        return df

    def _combine_asset_data(self, asset_dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple asset dataframes into unified dataset."""
        tprint_info("Combining multi-asset data...")
        
        combined_dfs = []
        for asset, df in asset_dataframes.items():
            # Add asset-specific features
            df_with_features = self._add_asset_features(df, asset)
            df_normalized = self._normalize_volatility_per_asset(df_with_features, asset)
            combined_dfs.append(df_normalized)
        
        # Combine all dataframes
        combined_df = pd.concat(combined_dfs, ignore_index=False)
        
        # Sort by timestamp to maintain temporal order
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp')
        elif combined_df.index.name == 'timestamp' or isinstance(combined_df.index, pd.DatetimeIndex):
            combined_df = combined_df.sort_index()
        
        tprint_success(f"✅ Combined {len(asset_dataframes)} assets into {len(combined_df)} total rows")
        
        # Print asset statistics
        tprint_info("Asset Statistics:")
        for asset, stats in self.asset_stats.items():
            tprint_info(f"  {asset}: vol_mean={stats['vol_mean']:.6f}, returns_mean={stats['returns_mean']:.6f}")
        
        return combined_df

    def _create_single_asset_config(self, global_config: Dict[str, Any], primary_asset: str) -> Dict[str, Any]:
        """Create a single-asset configuration for the underlying step."""
        single_config = global_config.copy()
        
        # Set symbol to primary asset for compatibility
        single_config['symbol'] = f"{primary_asset}USDT"
        
        # Add multi-asset context
        single_config['multi_asset_mode'] = True
        single_config['all_assets'] = global_config.get('assets', [])
        single_config['asset_stats'] = self.asset_stats
        
        return single_config

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the global multi-asset meta-labeling pipeline.
        
        Args:
            config: Configuration dictionary with assets list
            
        Returns:
            Execution result with combined model and asset-specific components
        """
        outcomes_dir = Path(config.get("outcomes_dir", "outcomes"))
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Extract assets from configuration
            assets = config.get('assets', [])
            if not assets or len(assets) < 2:
                raise ValueError("At least 2 assets required for multi-asset training")
            
            tprint_info(f"🌍 Starting Global Multi-Asset Meta-Labeling HPO")
            tprint_info(f"Assets: {', '.join(assets)}")
            tprint_info(f"Exchange: {config.get('exchange', 'binance')}")
            tprint_info(f"Timeframe: {config.get('timeframe', '15m')}")
            tprint_info(f"Execution Mode: {config.get('execution_mode', 'light')}")
            
            # Load data for all assets
            tprint_info("Phase 1: Loading multi-asset data...")
            asset_dataframes = {}
            
            for asset in assets:
                try:
                    asset_df = self._load_asset_data(config, asset)
                    asset_dataframes[asset] = asset_df
                except Exception as e:
                    tprint_error(f"❌ Failed to load {asset} data: {e}")
                    return {"success": False, "error": f"Failed to load {asset} data: {e}"}
            
            # Combine asset data with asset-specific features
            tprint_info("Phase 2: Combining assets with asset-specific features...")
            combined_data = self._combine_asset_data(asset_dataframes)
            self.combined_data = combined_data
            
            # Create single-asset configuration for underlying step
            primary_asset = assets[0]  # Use first asset as primary
            single_asset_config = self._create_single_asset_config(config, primary_asset)

            # Provide cross-asset context to downstream layers
            single_asset_config['assets'] = assets
            single_asset_config['cross_asset_data'] = asset_dataframes
            
            # Inject combined data into configuration
            single_asset_config['market_data'] = combined_data
            
            # Run the underlying meta-labeling HPO step on combined data
            tprint_info("Phase 3: Running meta-labeling HPO on combined dataset...")
            underlying_step = MetaLabelingHPOSampleWeightedStep("meta_labeling_hpo_sample_weighted")
            
            result = await underlying_step.execute(single_asset_config)
            
            if not result.get('success', False):
                tprint_error("❌ Underlying meta-labeling HPO failed")
                return result
            
            # Add multi-asset specific results
            result['multi_asset'] = {
                'assets': assets,
                'asset_stats': self.asset_stats,
                'combined_rows': len(combined_data),
                'primary_asset': primary_asset,
                'execution_mode': config.get('multi_asset_mode', 'global')
            }
            
            # Save multi-asset metadata
            metadata_file = outcomes_dir / f"global_meta_labeling_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metadata_file, 'w') as f:
                json.dump(result['multi_asset'], f, indent=2, default=str)
            
            result['metadata_file'] = str(metadata_file)
            
            tprint_success("🌍 Global Multi-Asset Meta-Labeling HPO completed successfully")
            tprint_info(f"Results saved to: {metadata_file}")
            
            return result
            
        except Exception as e:
            error_msg = f"Global multi-asset meta-labeling failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            import traceback
            tprint_error(traceback.format_exc())
            return {"success": False, "error": error_msg}


def register_global_meta_labeling_hpo_sample_weighted_step() -> None:
    """Register the global meta-labeling HPO sample weighted step in the registry."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("global_meta_labeling_hpo_sample_weighted", GlobalMetaLabelingHPOSampleWeightedStep)
    
    tprint_success("✅ Global meta-labeling HPO sample weighted step registered")


# Auto-register the step
register_global_meta_labeling_hpo_sample_weighted_step()
