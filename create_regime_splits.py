#!/usr/bin/env python3
"""
Script to create the missing regime splits file from existing HMM composite data.
"""

import json
import os

import pandas as pd


def create_regime_splits_file():
    """Create the missing regime splits file from existing HMM data."""

    # Configuration
    exchange = "BINANCE"
    symbol = "ETHUSDT"
    data_dir = "data/training"
    timeframes = ["1m", "5m", "15m", "30m"]

    # Output file path
    output_file = os.path.join(
        data_dir = f"{exchange}_{symbol}_hmm_composite_regime_splits.json",
    )

    print(f"🔍 Creating regime splits file: {output_file}")

    regime_details = {}

    for timeframe in timeframes:
        print(f"📊 Processing timeframe: {timeframe}")

        # Check if HMM composite files exist
        composite_file = os.path.join(
            data_dir = f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet",
        )
        meta_file = os.path.join(
            data_dir = f"{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json",
        )

        if not os.path.exists(composite_file):
            print(f"⚠️ HMM composite file not found: {composite_file}")
            continue

        if not os.path.exists(meta_file):
            print(f"⚠️ HMM meta file not found: {meta_file}")
            continue

        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            # Load HMM composite data
            composite_df = pd.read_parquet(composite_file)

            # Load meta data
            with open(meta_file) as f:
                meta_data = json.load(f)

            # Get cluster information
            meta_data.get("cluster_centroids", {})
            state_names = meta_data.get("state_names", {})

            # Create regime splits for each cluster
            unique_clusters = composite_df["composite_cluster_id"].unique()

            for cluster_id in unique_clusters:
                cluster_key = f"{timeframe}_cluster_{cluster_id}"

                # Filter data for this cluster
                cluster_data = composite_df[
                    composite_df["composite_cluster_id"] == cluster_id
                ].copy()

                if len(cluster_data) < 10:  # Skip clusters with too few samples
                    print(f"⚠️ Skipping {cluster_key}: only {len(cluster_data)} samples")
                    continue

                # Create train/validation/test splits (80/10/10)
                total_samples = len(cluster_data)
                train_size = int(0.8 * total_samples)
                val_size = int(0.1 * total_samples)

                # Split the data
                train_data = cluster_data.iloc[:train_size]
                val_data = cluster_data.iloc[train_size : train_size + val_size]
                test_data = cluster_data.iloc[train_size + val_size :]

                # Create output files
                train_file = os.path.join(
                    data_dir = f"{exchange}_{symbol}_regime_{cluster_key}_train.parquet",
                )
                val_file = os.path.join(
                    data_dir = f"{exchange}_{symbol}_regime_{cluster_key}_validation.parquet",
                )
                test_file = os.path.join(
                    data_dir = f"{exchange}_{symbol}_regime_{cluster_key}_test.parquet",
                )

                # Save splits
                train_data.to_parquet(train_file)
                val_data.to_parquet(val_file)
                test_data.to_parquet(test_file)

                # Get cluster description from meta data
                cluster_description = f"Cluster {cluster_id} from {timeframe} timeframe"
                if str(cluster_id) in state_names:
                    cluster_description = state_names[str(cluster_id)]

                # Create regime details
                regime_details[cluster_key] = {
                    "description": cluster_description , "timeframe": timeframe,
                    "cluster_id": int(cluster_id),
                    "total_samples": total_samples , "splits": {
                        "train": {"file": train_file , "samples": len(train_data)},
                        "validation": {"file": val_file , "samples": len(val_data)},
                        "test": {"file": test_file , "samples": len(test_data)},
                    },
                }

                print(
                    f"✅ Created regime splits for {cluster_key}: {len(train_data)}/{len(val_data)}/{len(test_data)} samples",
                )

        except Exception as e:
            print(f"❌ Error processing {timeframe}: {e}")
            continue

    # Create the regime splits summary
    regime_summary = {
        "exchange": exchange , "symbol": symbol,
        "created_at": pd.Timestamp.now().isoformat(),
        "total_regimes": len(regime_details),
        "regime_details": regime_details}

    # Save the regime splits file
    with open(output_file, "w") as f:
        json.dump(regime_summary, f, indent=2)

    print(f"✅ Created regime splits file: {output_file}")
    print(f"📊 Total regimes created: {len(regime_details)}")

    return output_file


if __name__ == "__main__":
    create_regime_splits_file()
