#!/usr/bin/env python3
import json

def debug_json_structure():
    """Debug the JSON structure to understand the issue."""
    with open("/Users/remyroche/Documents/Ares/artifacts/hmm_regime_unified_artifacts.json", 'r') as f:
        data = json.load(f)

    print("Top level keys:")
    for key in data.keys():
        print(f"  - {key}")

    print("\nChecking for regime_distribution_analysis:")
    if 'regime_distribution_analysis' in data:
        print("  ✅ Found regime_distribution_analysis")
        regime_dist = data['regime_distribution_analysis']
        print(f"  📊 Type: {type(regime_dist)}")
        print(f"  📊 Length: {len(regime_dist) if hasattr(regime_dist, '__len__') else 'N/A'}")

        if isinstance(regime_dist, dict):
            print("  📊 Keys in regime_distribution_analysis:")
            for key in regime_dist.keys():
                print(f"    - {key}")
                if key.startswith('regime_'):
                    regime_data = regime_dist[key]
                    print(f"      └─ Has indicator_averages: {'indicator_averages' in regime_data}")
                    if 'indicator_averages' in regime_data:
                        indicators = regime_data['indicator_averages']
                        print(f"        └─ Indicators count: {len(indicators)}")
                        print(f"        └─ Sample indicators: {list(indicators.keys())[:5]}")
    else:
        print("  ❌ regime_distribution_analysis NOT found")

if __name__ == "__main__":
    debug_json_structure()
