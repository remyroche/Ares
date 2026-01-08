#!/usr/bin/env python3
"""
Script to combine individual specialist outputs and apply GMM enhancement.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime

# Import GMM enhancement
from src.training.steps.market_analysis.gmm_enhanced_features import EnhancedGMMFeatures
from src.utils.tprint import tprint_info, tprint_success, tprint_error, tprint_warning

def load_specialist_outputs():
    """Load outputs from individual specialist training runs."""
    specialist_outputs = {}

    # List of specialists that should have outputs
    specialists = [
        "enhanced_ml_momentum_persistence_step",
        "enhanced_ml_smc_regime_step",
        "enhanced_ml_volatility_burst_step",
        "enhanced_ml_volume_force_step",
        "enhanced_xgb_macro_regime_step",
        "enhanced_xgb_meso_regime_step",
        "enhanced_ml_liquidity_regime_step",
        "enhanced_ml_path_regime_step",
        "enhanced_ml_risk_regime_step",
        "enhanced_ml_microstructure_step",
        "enhanced_ml_spectral_step"
    ]

    # Look for specialist outputs in versioned artifacts
    artifacts_dir = Path("versioned_artifacts")

    for specialist in specialists:
        # Try different possible paths
        possible_paths = [
            artifacts_dir / f"ETHUSDT_binance_15m_long_{specialist}",
            artifacts_dir / f"UNKNOWN_binance_15m_long_{specialist}",
        ]

        found_output = False
        for path in possible_paths:
            if path.exists():
                # Look for HDF5 files or other output files
                hdf5_files = list(path.glob("*.h5"))
                json_files = list(path.glob("*.json"))

                if hdf5_files:
                    try:
                        # Load the most recent HDF5 file
                        hdf5_file = sorted(hdf5_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                        data = pd.read_hdf(hdf5_file)
                        specialist_outputs[specialist] = data
                        tprint_success(f"✅ Loaded {specialist} from {hdf5_file}")
                        found_output = True
                        break
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to load HDF5 from {path}: {e}")

                if json_files and not found_output:
                    # Try JSON metadata
                    try:
                        metadata_file = path / "metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            tprint_info(f"📄 Found metadata for {specialist}: {len(metadata.get('versions', []))} versions")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to read metadata from {path}: {e}")

        if not found_output:
            tprint_warning(f"⚠️ No output found for {specialist}")

    return specialist_outputs

def combine_specialist_outputs(specialist_outputs):
    """Combine outputs from multiple specialists."""
    if not specialist_outputs:
        tprint_error("❌ No specialist outputs to combine")
        return None

    tprint_info(f"🔗 Combining outputs from {len(specialist_outputs)} specialists...")

    # For now, concatenate all outputs horizontally
    # In a real implementation, you'd want to align them properly by timestamp
    combined_data = None

    for specialist_name, output_df in specialist_outputs.items():
        if output_df is not None and not output_df.empty:
            if combined_data is None:
                combined_data = output_df.copy()
                combined_data.columns = [f"{specialist_name}_{col}" for col in combined_data.columns]
            else:
                # Align by index (timestamp)
                output_df_aligned = output_df.reindex(combined_data.index)
                output_df_aligned.columns = [f"{specialist_name}_{col}" for col in output_df.columns]
                combined_data = pd.concat([combined_data, output_df_aligned], axis=1)

    if combined_data is not None:
        # Remove rows with all NaN values
        combined_data = combined_data.dropna(how='all')
        tprint_success(f"✅ Combined data shape: {combined_data.shape}")

    return combined_data

def apply_gmm_enhancement(combined_data):
    """Apply GMM enhancement to the combined specialist outputs."""
    if combined_data is None or combined_data.empty:
        tprint_error("❌ No data to enhance")
        return None

    tprint_info("🎯 Applying GMM enhancement...")

    try:
        # Initialize GMM enhancer
        gmm_enhancer = EnhancedGMMFeatures()

        # Apply enhancement
        enhanced_features = gmm_enhancer.fit_transform(combined_data)

        if enhanced_features is not None and not enhanced_features.empty:
            tprint_success(f"✅ GMM enhancement completed: {enhanced_features.shape}")
            return enhanced_features
        else:
            tprint_error("❌ GMM enhancement returned empty result")
            return None

    except Exception as e:
        tprint_error(f"❌ GMM enhancement failed: {e}")
        import traceback
        tprint_error(f"Traceback: {traceback.format_exc()}")
        return None

def save_results(enhanced_features, specialist_outputs):
    """Save the enhanced features and metadata."""
    if enhanced_features is None or enhanced_features.empty:
        tprint_error("❌ No enhanced features to save")
        return

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outcomes/specialists_with_gmm_ETHUSDT_15m_long_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save enhanced features
    output_file = output_dir / "enhanced_features.parquet"
    enhanced_features.to_parquet(output_file)
    tprint_success(f"💾 Saved enhanced features to {output_file}")

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "symbol": "ETHUSDT",
        "timeframe": "15m",
        "direction": "long",
        "n_specialists": len(specialist_outputs),
        "enhanced_features_shape": list(enhanced_features.shape),
        "specialist_names": list(specialist_outputs.keys()),
        "feature_columns": list(enhanced_features.columns)
    }

    metadata_file = output_dir / "training_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    tprint_success(f"💾 Saved metadata to {metadata_file}")

def main():
    """Main execution."""
    tprint_info("🎯 Starting specialist output combination and GMM enhancement...")

    # Load individual specialist outputs
    specialist_outputs = load_specialist_outputs()

    if not specialist_outputs:
        tprint_error("❌ No specialist outputs found. Please run individual specialists first.")
        return 1

    # Combine outputs
    combined_data = combine_specialist_outputs(specialist_outputs)

    if combined_data is None:
        tprint_error("❌ Failed to combine specialist outputs")
        return 1

    # Apply GMM enhancement
    enhanced_features = apply_gmm_enhancement(combined_data)

    if enhanced_features is None:
        tprint_error("❌ Failed to apply GMM enhancement")
        return 1

    # Save results
    save_results(enhanced_features, specialist_outputs)

    tprint_success("🎉 Specialist combination and GMM enhancement completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())