#!/usr/bin/env python3
"""
Run base training step only, using existing labels
"""
import sys
import os
import argparse

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from extreme_price_movements.utils import tprint
from extreme_price_movements.training import train_models_from_artifacts
from extreme_price_movements.config import CFG
from extreme_price_movements.data_store import load_features
import pandas as pd

def run_base_training_only(ts_sig=None, force=False):
    """Run base training step using existing artifacts."""
    
    if ts_sig is None:
        # Find latest feature timestamp
        from extreme_price_movements.run_pipeline import _find_latest_feature_ts
        ts_sig = _find_latest_feature_ts("../data")
        if ts_sig is None:
            tprint("ERROR: No feature directories found. Run feature_generation first.")
            return False
    
    tprint(f"Running base training only for ts_sig={ts_sig}")
    
    # Load configuration
    cfg = CFG.copy()
    
    # Load features
    try:
        feats = load_features(ts_sig, "../data")
        if feats is None or len(feats) == 0:
            tprint("ERROR: Features not found.")
            return False
        tprint(f"Loaded {len(feats)} feature matrices")
    except Exception as e:
        tprint(f"ERROR: Failed to load features: {e}")
        return False
    
    # Prepare datasets for training
    try:
        # This will use existing labels from the artifacts directory
        from extreme_price_movements.training import generate_label_datasets
        
        # Get training universe
        from extreme_price_movements.universe import get_training_universe
        from extreme_price_movements.data_store import PartitionedOHLCVStore
        
        store = PartitionedOHLCVStore(root_dir="../data", timeframe=cfg["timeframe"])
        train_syms = get_training_universe(None, cfg, store, ts_sig=ts_sig)
        
        tprint(f"Training universe: {len(train_syms)} symbols")
        
        # Generate label datasets (this will use existing cached labels if available)
        datasets = generate_label_datasets(ts_sig, train_syms, cfg, store)
        
        if not datasets:
            tprint("ERROR: Failed to generate label datasets")
            return False
            
        tprint(f"Generated datasets for {len(datasets)} geometries")
        
    except Exception as e:
        tprint(f"ERROR: Failed to prepare datasets: {e}")
        return False
    
    # Run base model training
    try:
        tprint("Starting base model training...")
        trained_bundle = train_models_from_artifacts(datasets, cfg)
        
        if trained_bundle:
            tprint("✅ Base training completed successfully!")
            return True
        else:
            tprint("❌ Base training failed")
            return False
            
    except Exception as e:
        tprint(f"ERROR: Base training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Run base training only")
    parser.add_argument("--ts", help="Timestamp override (YYYYMMDD_HHMMSS)")
    parser.add_argument("--force", action="store_true", help="Force recompute")
    args = parser.parse_args()
    
    # Parse timestamp if provided
    ts_sig = None
    if args.ts:
        try:
            ts_sig = pd.to_datetime(args.ts, format="%Y%m%d_%H%M%S").tz_localize("UTC")
        except ValueError:
            ts_sig = pd.Timestamp(args.ts).tz_localize("UTC")
    
    success = run_base_training_only(ts_sig=ts_sig, force=args.force)
    
    if success:
        tprint("🎉 Base training completed successfully!")
    else:
        tprint("❌ Base training failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
