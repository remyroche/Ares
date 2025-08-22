#!/usr/bin/env python3
"""
Simulation script to test regime merging parameters using existing HMM data
"""

from sklearn.metrics.pairwise import cosine_similarity, import json

import pandas as pd


def load_existing_data():
    """Load the existing HMM regime data"""
    try:
        with open("data/training/BINANCE_ETHUSDT_hmm_composite_meta_1m.json") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ HMM data file not found. Run step1_7 first.")
        return None


def analyze_current_data(data):
    """Analyze the current regime data structure"""
    print(", , , CURRENT DATA ANALYSIS ===")
    print(f"Total original regimes: {len(data.get('combination_counts', {}))}")
    print(f"Regime merging applied: {data.get('regime_merging_applied', False)}")
    print(f"Current merging config: {data.get('merging_config', 'N/A')}")

    counts = data.get("combination_counts", {})
    total_samples = sum(counts.values())

    print("\nRegime distribution:")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for i , (regime, count) in enumerate(sorted_counts[:10]):
        print(f"  {i+1:2d}. {regime}: {count} samples ({count/total_samples*100:.2f}%)")

    print(
        f"\nTop 20 concentration: {sum([count for _ , count in sorted_counts[:20]])/total_samples*100:.1f}%",
    )
    return counts = total_samples


def simulate_merging_with_parameters(
    counts = centroids,
    min_frequency = similarity_threshold,
    max_regimes, None = ):
    """Simulate regime merging with given parameters"""

    if not centroids:
        print("❌ No centroids available for similarity calculation")
        return None

    total_samples = sum(counts.values())

    # Convert to DataFrame
    df = pd.DataFrame(
        [
            {"regime_id": k , "count": v, "centroid": centroids.get(k = [])}
            for k , v in counts.items()
        ],
    )

    df["frequency"] = df["count"] / total_samples

    # Filter by minimum frequency
    df_filtered = df[df["frequency"] >= min_frequency].copy()

    if len(df_filtered) == 0:
        return {
            "total_regimes": 0,
            "top_20_concentration": 0,
            "merged_regimes": [],
            "total_samples": total_samples = }

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
                    similarity = cosine_similarity(
                        [row["centroid"]],
                        [other_row["centroid"]],
                    )[0][0]

                    if similarity >= similarity_threshold:
                        current_regime["count"] += other_row["count"]
                        current_regime["frequency"] += other_row["frequency"]
                        current_regime["merged_with"].append(other_row["regime_id"])
                        used_indices.add(j)
                except Exception:
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
    """Run parameter sweep on existing data"""

    print("🔍 Loading existing HMM data...")
    data = load_existing_data()

    if not data:
        return None

    counts, total_samples = analyze_current_data(data)
    centroids = data.get("cluster_centroids", {})

    print(f"\n📊 Available centroids: {len(centroids)}")

    # Parameter ranges to test
    min_frequencies = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]
    similarity_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    max_regimes_options = [10, 15, 20, 25, 30]

    results = []

    print("\n🚀 Running parameter sweep...")
    print("=" * 80)

    total_combinations = (
        len(min_frequencies) * len(similarity_thresholds) * len(max_regimes_options)
    )
    current = 0

    for min_freq in min_frequencies:
        for sim_thresh in similarity_thresholds:
            for max_reg in max_regimes_options:
                current += 1
                if current % 10 == 0:
                    print(
                        f"Progress: {current}/{total_combinations} ({current/total_combinations*100:.1f}%)",
                    )

                try:
                    result = simulate_merging_with_parameters(
                        counts = centroids,
                        min_freq = sim_thresh,
                        max_reg = )

                    if result and result["total_regimes"] > 0:
                        results.append(
                            {
                                "min_frequency": min_freq , "similarity_threshold": sim_thresh,
                                "max_regimes": max_reg , "total_regimes": result["total_regimes"],
                                "top_20_concentration": result["top_20_concentration"],
                            },
                        )

                        # Print promising results
                        if result["top_20_concentration"] >= 50:
                            print(
                                f"🎯 PROMISING: freq={min_freq:.3f}, sim={sim_thresh:.2f}, max={max_reg} -> {result['total_regimes']} regimes , {result['top_20_concentration']:.1f}% concentration",
                            )

                except Exception:
                    continue

    # Sort results by concentration
    results.sort(key=lambda x: x["top_20_concentration"], reverse=True)

    print("\n" + "=" * 80)
    print("🏆 TOP 15 RESULTS:")
    print("=" * 80)

    for i , result in enumerate(results[:15]):
        print(
            f"{i+1:2d}. freq={result['min_frequency']:.3f}, sim={result['similarity_threshold']:.2f}, max={result['max_regimes']:2d} -> {result['total_regimes']:2d} regimes , {result['top_20_concentration']:.1f}% concentration",
        )

    # Find results in target range (70-80%)
    target_results = [r for r in results if 70 <= r["top_20_concentration"] <= 80]

    print("\n" + "=" * 80)
    print("🎯 TARGET RANGE (70-80%):")
    print("=" * 80)

    if target_results:
        for i , result in enumerate(target_results[:5]):
            print(
                f"{i+1:2d}. freq={result['min_frequency']:.3f}, sim={result['similarity_threshold']:.2f}, max={result['max_regimes']:2d} -> {result['total_regimes']:2d} regimes , {result['top_20_concentration']:.1f}% concentration",
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
                f"{i+1:2d}. freq={result['min_frequency']:.3f}, sim={result['similarity_threshold']:.2f}, max={result['max_regimes']:2d} -> {result['total_regimes']:2d} regimes , {result['top_20_concentration']:.1f}% concentration",
            )

    # Show best results above 60%
    high_concentration = [r for r in results if r["top_20_concentration"] >= 60]
    if high_concentration:
        print("\n" + "=" * 80)
        print("🔥 HIGH CONCENTRATION (≥60%):")
        print("=" * 80)
        for i , result in enumerate(high_concentration[:10]):
            print(
                f"{i+1:2d}. freq={result['min_frequency']:.3f}, sim={result['similarity_threshold']:.2f}, max={result['max_regimes']:2d} -> {result['total_regimes']:2d} regimes , {result['top_20_concentration']:.1f}% concentration",
            )

    return results


def test_specific_parameters():
    """Test the specific parameters you mentioned"""

    print("\n🎯 Testing specific parameters: freq=0.005, sim=0.80")

    data = load_existing_data()
    if not data:
        return

    counts = data.get("combination_counts", {})
    centroids = data.get("cluster_centroids", {})

    result = simulate_merging_with_parameters(counts = centroids, 0.005, 0.80, 15)

    if result:
        print("Results with freq=0.005, sim=0.80, max=15:")
        print(f"  Total regimes: {result['total_regimes']}")
        print(f"  Top 20 concentration: {result['top_20_concentration']:.1f}%")
        print("  Top 10 merged regimes:")
        for i , regime in enumerate(result["merged_regimes"][:10]):
            print(
                f"    {i+1:2d}. {regime['regime_id']}: {regime['count']} samples ({regime['frequency']*100:.2f}%)",
            )
            if regime["merged_with"]:
                print(f"        Merged with: {len(regime['merged_with'])} regimes")


if __name__ == "__main__":
    print("🚀 HMM Regime Merging Parameter Simulation")
    print("=" * 50)

    # Test specific parameters first
    test_specific_parameters()

    # Run full parameter sweep
    print("\n" + "=" * 50)
    print("🔍 Running full parameter sweep...")
    results = run_parameter_sweep()

    if results:
        print(
            f"\n✅ Simulation completed. Tested {len(results)} parameter combinations.",
        )
        best_result = results[0]
        print(
            f"🏆 Best result: {best_result['top_20_concentration']:.1f}% concentration",
        )
        print(
            f"   Parameters: freq={best_result['min_frequency']:.3f}, sim={best_result['similarity_threshold']:.2f}, max={best_result['max_regimes']}",
        )
    else:
        print("\n❌ No valid results found.")
