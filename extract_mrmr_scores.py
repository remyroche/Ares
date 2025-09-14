#!/usr/bin/env python3
"""
Script to extract detailed mRMR and relevance scores for all features.

This script loads the feature selection data and runs mRMR analysis to show
detailed scores for all features, not just the selected ones.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from src.training.utils.feature_selection.selection_methods import MRMRSelector
    from src.training.utils.feature_selection.data_validation import DataValidator
    from src.utils.tprint import tprint
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

logger = get_logger("MRMR_Extractor")

def load_feature_data():
    """Load feature data from the data cache."""
    try:
        # Look for feature data in data_cache
        data_cache_dir = Path("data_cache")

        # Try to find recent feature data files
        feature_files = list(data_cache_dir.glob("*features*.parquet"))
        feature_files.extend(list(data_cache_dir.glob("*features*.pkl")))

        if not feature_files:
            # Try broader search
            all_parquet = list(data_cache_dir.glob("*.parquet"))
            feature_files = [f for f in all_parquet if 'feature' in f.name.lower()]

        if not feature_files:
            logger.error("❌ No feature data files found in data_cache")
            return None, None

        # Use the most recent feature file
        feature_file = max(feature_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"📂 Loading features from: {feature_file}")

        if feature_file.suffix == '.parquet':
            df = pd.read_parquet(feature_file)
        else:
            df = pd.read_pickle(feature_file)

        logger.info(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Separate features and target
        # Assume the last column is the target or look for common target names
        target_columns = ['target', 'label', 'y', 'close_return', 'returns']
        target_col = None

        for col in target_columns:
            if col in df.columns:
                target_col = col
                break

        if target_col is None:
            # Use the last column as target
            target_col = df.columns[-1]
            logger.warning(f"⚠️ Using last column '{target_col}' as target")

        # Get feature columns (exclude target)
        feature_cols = [col for col in df.columns if col != target_col]

        X = df[feature_cols].values
        y = df[target_col].values

        # Handle NaN values
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            logger.warning(f"⚠️ Found {np.sum(nan_mask)} NaN values, filling with 0")
            X = np.nan_to_num(X, nan=0.0)

        logger.info(f"📊 Feature matrix: {X.shape}, Target: {y.shape}")

        return X, y, feature_cols

    except Exception as e:
        logger.error(f"❌ Failed to load feature data: {e}")
        return None, None, None

def extract_mrmr_scores(X, y, feature_names, n_features=90):
    """Extract detailed mRMR and relevance scores for all features."""

    try:
        # Initialize MRMR selector
        mrmr_config = {
            'relevance_method': 'mutual_info',
            'redundancy_method': 'correlation'
        }

        selector = MRMRSelector(config=mrmr_config)

        # Calculate relevance scores for ALL features
        relevance_scores = selector._calculate_relevance_scores(X, y, feature_names)

        # Sort features by relevance score (descending)
        sorted_features = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

        logger.info("🔍 MRMR Feature Analysis Results:")
        logger.info("=" * 80)
        logger.info(f"{'Rank':<4} {'Feature Name':<25} {'Relevance':<10} {'Redundancy':<10} {'mRMR Score':<10}")
        logger.info("-" * 80)

        # Calculate mRMR scores for all features
        selected_features = []
        all_scores = []

        for rank, (feature_idx, relevance) in enumerate(sorted_features):
            feature_name = feature_names[feature_idx]

            # Calculate redundancy for this feature (with previously selected features)
            redundancy = selector._calculate_redundancy(feature_idx, selected_features, X)
            mrmr_score = relevance - redundancy

            all_scores.append({
                'rank': rank + 1,
                'feature_name': feature_name,
                'relevance': relevance,
                'redundancy': redundancy,
                'mrmr_score': mrmr_score,
                'feature_idx': feature_idx
            })

            logger.info(f"{rank+1:<4} {feature_name:<25} {relevance:<10.4f} {redundancy:<10.4f} {mrmr_score:<10.4f}")

            # Add to selected features for next redundancy calculation
            selected_features.append(feature_idx)

        # Now show the top 90 features that would be selected
        logger.info("\n🎯 Top 90 Features Selected by mRMR:")
        logger.info("=" * 80)

        for i in range(min(90, len(all_scores))):
            score = all_scores[i]
            logger.info(f"{score['rank']:2d}. {score['feature_name']} (mRMR: {score['mrmr_score']:.4f}, relevance: {score['relevance']:.4f})")

        return all_scores

    except Exception as e:
        logger.error(f"❌ Failed to extract mRMR scores: {e}")
        return None

def main():
    """Main function to extract and display mRMR scores."""
    tprint("🚀 Extracting detailed mRMR and relevance scores...\n")

    # Load data
    X, y, feature_names = load_feature_data()

    if X is None or y is None:
        tprint("❌ Failed to load data")
        return

    # Extract scores
    scores = extract_mrmr_scores(X, y, feature_names, n_features=90)

    if scores:
        tprint("✅ Analysis completed successfully")

        # Save detailed results to file
        output_file = "mrmr_detailed_scores.csv"
        try:
            df_scores = pd.DataFrame(scores)
            df_scores.to_csv(output_file, index=False)
            tprint(f"💾 Detailed scores saved to: {output_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

    else:
        tprint("❌ Failed to extract scores")

if __name__ == "__main__":
    main()
