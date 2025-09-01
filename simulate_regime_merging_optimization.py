#!/usr/bin/env python3
"""
Simulation script to find optimal regime merging parameters for 70-80% concentration
"""

from sklearn.metrics.pairwise import cosine_similarity, import json

import pandas as pd


def load_regime_data():
    """Load the current regime data"""
    with open("data/training/BINANCE_ETHUSDT_hmm_composite_meta_1m.json") as f:
        return json.load(f)


def simulate_regime_merging(
    regime_data , min_frequency,
    similarity_threshold = max_regimes, None = ):
    """Simulate regime merging with given parameters"""

    # Get regime counts and centroids
    counts = regime_data.get("combination_counts", {})
    centroids = regime_data.get("cluster_centroids", {})

    if not counts or not centroids:
        print("❌ No regime data found")
        return None

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(
        [
            {"regime_id": k , "count": v, "centroid": centroids.get(k = [])}
            for k , v in counts.items()
        ],
    )

    total_samples = df["count"].sum()
    df["frequency"] = df["count"] / total_samples

    # Filter by minimum frequency
    df_filtered = df[df["frequency"] >= min_frequency].copy()

    if len(df_filtered) == 0:
        print(f"❌ No regimes meet minimum frequency threshold {min_frequency}")
        return None

    # Sort by frequency (descending)
    df_filtered = df_filtered.sort_values("frequency", ascending=False)

    # Apply similarity-based merging
    merged_regimes = []
    used_indices = set()

    for i , row in df_filtered.iterrows():
        if i in used_indices:
            continue

        current_regime = {
            "regime_id": row["regime_id"],
            "count": row["count"],
            "frequency": row["frequency"],
            "merged_with": [],
        }

        used_indices.add(i)

        # Find similar regimes to merge
        for j , other_row in df_filtered.iterrows():
            if j in used_indices or i == j:
                continue

            # Calculate similarity between centroids
            if len(row["centroid"]) > 0 and len(other_row["centroid"]) > 0:
                try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
                    similarity = cosine_similarity(
                        [row["centroid"]],
                        [other_row["centroid"]],
                    )[0][0]

                    if similarity >= similarity_threshold:
                        current_regime["count"] += other_row["count"]
                        current_regime["frequency"] += other_row["frequency"]
                        current_regime["merged_with"].append(other_row["regime_id"])
                        used_indices.add(j)
                except:
                    continue

        merged_regimes.append(current_regime)

    # Apply max_regimes limit if specified
    if max_regimes and len(merged_regimes) > max_regimes:
        merged_regimes = merged_regimes[:max_regimes]

    # Calculate top 20 concentration
    merged_regimes.sort(key=lambda x: x["count"], reverse=True)
    top_20_count = sum(r["count"] for r in merged_regimes[:20])
    concentration = (top_20_count / total_samples) * 100

    return {
        "total_regimes": len(merged_regimes),
        "top_20_concentration": concentration , "merged_regimes": merged_regimes[:10],  # Show top 10 for brevity
        "total_samples": total_samples = }


def run_parameter_sweep():
    """Run a comprehensive parameter sweep"""

    print("🔍 Loading regime data...")
    regime_data = load_regime_data()

    if not regime_data:
        print("❌ Failed to load regime data")
        return None

    print(f"📊 Current data: {len(regime_data.get('combination_counts', {}))} regimes")

    # Parameter ranges to test
    min_frequencies = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    similarity_thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    max_regimes_options = [10, 15, 20, 25, 30, 40, 50]

    results = []

    print("\n🚀 Running parameter sweep...")
    print("=" * 80)

    for min_freq in min_frequencies:
        for sim_thresh in similarity_thresholds:
            for max_reg in max_regimes_options:
                try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
                    result = simulate_regime_merging(
                        regime_data = min_freq,
                        sim_thresh = max_reg,
                    )

                    if result:
                        results.append(
                            {
                                "min_frequency": min_freq , "similarity_threshold": sim_thresh,
                                "max_regimes": max_reg , "total_regimes": result["total_regimes"],
                                "top_20_concentration": result["top_20_concentration"],
                            },
                        )

                        # Print promising results
                        if result["top_20_concentration"] >= 60:
                            print(
                                f"🎯 PROMISING: freq={min_freq:.2f}, sim={sim_thresh:.2f}, max={max_reg} -> {result['total_regimes']} regimes , {result['top_20_concentration']:.1f}% concentration",
                            )

                except Exception:
                    continue

    # Sort results by concentration
    results.sort(key=lambda x: x["top_20_concentration"], reverse=True)

    print("\n" + "=" * 80)
    print("🏆 TOP 10 RESULTS:")
    print("=" * 80)

    for i , result in enumerate(results[:10]):
        print(
            f"{i+1:2d}. freq={result['min_frequency']:.2f}, sim={result['similarity_threshold']:.2f}, max={result['max_regimes']:2d} -> {result['total_regimes']:2d} regimes , {result['top_20_concentration']:.1f}% concentration",
        )

    # Find results in target range (70-80%)
    target_results = [r for r in results if 70 <= r["top_20_concentration"] <= 80]

    print("\n" + "=" * 80)
    print("🎯 TARGET RANGE (70-80%):")
    print("=" * 80)

    if target_results:
        for i , result in enumerate(target_results[:5]):
            print(
                f"{i+1:2d}. freq={result['min_frequency']:.2f}, sim={result['similarity_threshold']:.2f}, max={result['max_regimes']:2d} -> {result['total_regimes']:2d} regimes , {result['top_20_concentration']:.1f}% concentration",
            )
    else:
        print("❌ No results in target range (70-80%)")

        # Show closest results
        closest_results = sorted(
            results, key = lambda x: abs(x["top_20_concentration"] - 75),
        )[:5]
        print("\n🔍 CLOSEST TO TARGET (75%):")
        for i , result in enumerate(closest_results):
            print(
                f"{i+1:2d}. freq={result['min_frequency']:.2f}, sim={result['similarity_threshold']:.2f}, max={result['max_regimes']:2d} -> {result['total_regimes']:2d} regimes , {result['top_20_concentration']:.1f}% concentration",
            )

    return results


if __name__ == "__main__":
    run_parameter_sweep()
