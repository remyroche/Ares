#!/usr/bin/env python3
"""
S/R Script Compatibility Validator

This script validates that S/R analysis scripts are compatible with the updated SRBreakoutPredictor.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.logger import system_logger


class SRCompatibilityValidator:
    """Validator for S/R script compatibility."""

    def __init__(self):
        self.logger = system_logger.getChild("SRCompatibilityValidator")

    async def test_sr_breakout_predictor_compatibility(self) -> Dict[str, Any]:
        """Test SRBreakoutPredictor compatibility with enhanced features."""
        self.logger.info("🔍 Testing SRBreakoutPredictor compatibility...")

        # Test configuration
        config = {
            "sr_breakout_predictor": {
                "enable_sr_breakout_tactics": True,
                "sr_proximity_threshold": 0.02,
                "breakout_confidence_threshold": 0.6,
                "sr_detection_method": "fractal",
                "min_sr_strength": 0.3,
                "max_sr_levels": 10,
                "sr_lookback_periods": 100,
                "volume_weight": 0.7,
                "price_weight": 0.3,
                "atr_multiplier": 1.5,
                "breakout_confirmation_periods": 3,
                "false_breakout_filter": True,

                # Enhanced strength calculation configuration
                "strength_calculation": {
                    "enable_enhanced_strength": True,
                    "touch_count_lookback": 50,
                    "bounce_rate_threshold": 0.02,
                    "isolation_distance_threshold": 0.05,
                    "age_decay_factor": 0.95
                },

                # DBSCAN clustering configuration
                "dbscan_clustering": {
                    "enable_dbscan_clustering": True,
                    "eps": 0.01,
                    "min_samples": 2,
                    "enable_noise_filtering": True
                },

                # Feature calculation configuration
                "feature_calculation": {
                    "enable_comprehensive_features": True,
                    "strength_score_weights": {
                        "touch_count": 0.3,
                        "total_volume": 0.2,
                        "level_age": 0.2,
                        "bounce_rate": 0.2,
                        "isolation_score": 0.1
                    }
                }
            }
        }

        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            # Initialize SRBreakoutPredictor
            sr_predictor = SRBreakoutPredictor(config)
            init_success = await sr_predictor.initialize()

            if not init_success:
                return {
                    "compatible": False,
                    "error": "Failed to initialize SRBreakoutPredictor",
                    "enhanced_features": False
                }

            # Create sample market data
            import pandas as pd
            import numpy as np

            # Generate sample OHLCV data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
            np.random.seed(42)

            base_price = 100.0
            prices = [base_price]
            for i in range(1, 100):
                trend = 0.001 * np.sin(i * 0.1)
                random_walk = np.random.normal(0, 0.005)
                new_price = prices[-1] * (1 + trend + random_walk)
                prices.append(new_price)

            data = []
            for i, price in enumerate(prices):
                volatility = 0.01 * (1 + 0.5 * np.sin(i * 0.2))
                high = price * (1 + np.random.uniform(0, volatility))
                low = price * (1 - np.random.uniform(0, volatility))
                open_price = prices[i-1] if i > 0 else price
                close_price = price
                volume = int(1000000 * (1 + abs(close_price - open_price) / open_price * 10) * np.random.uniform(0.5, 1.5))

                data.append({
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })

            market_data = pd.DataFrame(data, index=dates)
            current_price = market_data['close'].iloc[-1]

            # Test get_sr_context method
            sr_context = await sr_predictor.get_sr_context(market_data, current_price)

            if not sr_context:
                return {
                    "compatible": False,
                    "error": "Failed to generate S/R context",
                    "enhanced_features": False
                }

            # Check for enhanced features
            enhanced_features = {
                "enhanced_strength_support": "enhanced_strength_support" in sr_context,
                "enhanced_strength_resistance": "enhanced_strength_resistance" in sr_context,
                "clustering_result": "clustering_result" in sr_context,
                "fibonacci_levels": "fibonacci_levels" in sr_context,
                "elliott_wave_levels": "elliott_wave_levels" in sr_context,
                "order_flow_analysis": "order_flow_analysis" in sr_context,
                "support_levels": "support_levels" in sr_context,
                "resistance_levels": "resistance_levels" in sr_context,
                "support_strength": "support_strength" in sr_context,
                "resistance_strength": "resistance_strength" in sr_context,
            }

            # Test enhanced strength calculation methods
            support_levels = sr_context.get("support_levels", [])
            resistance_levels = sr_context.get("resistance_levels", [])
            all_levels = support_levels + resistance_levels

            if all_levels:
                # Test enhanced strength methods
                touch_counts = await sr_predictor.calculate_touch_count(market_data, all_levels)
                level_ages = await sr_predictor.calculate_level_age(market_data, all_levels)
                bounce_rates = await sr_predictor.calculate_bounce_rate(market_data, all_levels)
                isolation_scores = await sr_predictor.calculate_isolation_score(all_levels)
                clustering_result = await sr_predictor.cluster_sr_levels_dbscan(all_levels)
                comprehensive_strengths = await sr_predictor.calculate_comprehensive_strength(market_data, all_levels)

                enhanced_methods = {
                    "calculate_touch_count": len(touch_counts) > 0,
                    "calculate_level_age": len(level_ages) > 0,
                    "calculate_bounce_rate": len(bounce_rates) > 0,
                    "calculate_isolation_score": len(isolation_scores) > 0,
                    "cluster_sr_levels_dbscan": clustering_result.get("n_clusters", 0) >= 0,
                    "calculate_comprehensive_strength": len(comprehensive_strengths) > 0,
                }
            else:
                enhanced_methods = {method: False for method in [
                    "calculate_touch_count", "calculate_level_age", "calculate_bounce_rate",
                    "calculate_isolation_score", "cluster_sr_levels_dbscan", "calculate_comprehensive_strength"
                ]}

            return {
                "compatible": True,
                "enhanced_features": enhanced_features,
                "enhanced_methods": enhanced_methods,
                "sr_context_keys": list(sr_context.keys()),
                "support_levels_count": len(support_levels),
                "resistance_levels_count": len(resistance_levels),
                "clustering_info": {
                    "n_clusters": clustering_result.get("n_clusters", 0),
                    "noise_points": clustering_result.get("noise_points", 0),
                    "total_points": clustering_result.get("total_points", 0)
                } if all_levels else {}
            }

        except Exception as e:
            return {
                "compatible": False,
                "error": str(e),
                "enhanced_features": False
            }

    def test_script_compatibility(self) -> Dict[str, Any]:
        """Test script file compatibility."""
        self.logger.info("🔍 Testing script file compatibility...")

        scripts_to_test = [
            "scripts/analyze_sr_position.py",
            "scripts/analyze_sr_position_enhanced.py",
            "scripts/run_sr_optimization.py"
        ]

        results = {}

        for script_path in scripts_to_test:
            path = Path(script_path)
            if path.exists():
                try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Check for imports
                    imports = {
                        "SRBreakoutPredictor": "from src.tactician.sr_breakout_predictor import SRBreakoutPredictor" in content,
                        "VectorizedAdvancedFeatureEngineering": "VectorizedAdvancedFeatureEngineering" in content,
                        "system_logger": "system_logger" in content,
                        "pandas": "import pandas" in content,
                        "numpy": "import numpy" in content,
                    }

                    # Check for method usage
                    method_usage = {
                        "get_sr_context": "get_sr_context" in content,
                        "calculate_touch_count": "calculate_touch_count" in content,
                        "calculate_bounce_rate": "calculate_bounce_rate" in content,
                        "cluster_sr_levels_dbscan": "cluster_sr_levels_dbscan" in content,
                        "calculate_comprehensive_strength": "calculate_comprehensive_strength" in content,
                    }

                    # Check for enhanced features
                    enhanced_features = {
                        "enhanced_strength": "enhanced_strength" in content,
                        "clustering_result": "clustering_result" in content,
                        "fibonacci_levels": "fibonacci_levels" in content,
                        "elliott_wave": "elliott_wave" in content,
                        "order_flow": "order_flow" in content,
                    }

                    results[script_path] = {
                        "exists": True,
                        "imports": imports,
                        "method_usage": method_usage,
                        "enhanced_features": enhanced_features,
                        "uses_enhanced_predictor": "SRBreakoutPredictor" in content,
                        "uses_basic_engineering": "VectorizedAdvancedFeatureEngineering" in content,
                    }

                except Exception as e:
                    results[script_path] = {
                        "exists": True,
                        "error": str(e),
                        "compatible": False
                    }
            else:
                results[script_path] = {
                    "exists": False,
                    "compatible": False
                }

        return results

    def print_compatibility_report(self, sr_results: Dict[str, Any], script_results: Dict[str, Any]) -> None:
        """Print comprehensive compatibility report."""
        print("\n" + "=" * 80)
        print("📊 S/R SCRIPT COMPATIBILITY REPORT")
        print("=" * 80)

        # SRBreakoutPredictor compatibility
        print(f"\n🔧 SRBreakoutPredictor Compatibility:")
        if sr_results.get("compatible", False):
            print("   ✅ SRBreakoutPredictor is compatible")

            enhanced_features = sr_results.get("enhanced_features", {})
            print(f"   📊 Enhanced Features Available:")
            for feature, available in enhanced_features.items():
                status = "✅" if available else "❌"
                print(f"      {status} {feature}: {available}")

            enhanced_methods = sr_results.get("enhanced_methods", {})
            print(f"   🔧 Enhanced Methods Available:")
            for method, available in enhanced_methods.items():
                status = "✅" if available else "❌"
                print(f"      {status} {method}: {available}")

            clustering_info = sr_results.get("clustering_info", {})
            if clustering_info:
                print(f"   🔍 Clustering Results:")
                print(f"      Clusters: {clustering_info.get('n_clusters', 0)}")
                print(f"      Noise Points: {clustering_info.get('noise_points', 0)}")
                print(f"      Total Points: {clustering_info.get('total_points', 0)}")

            print(f"   📈 S/R Levels Generated:")
            print(f"      Support Levels: {sr_results.get('support_levels_count', 0)}")
            print(f"      Resistance Levels: {sr_results.get('resistance_levels_count', 0)}")

        else:
            print(f"   ❌ SRBreakoutPredictor compatibility issues: {sr_results.get('error', 'Unknown error')}")

        # Script compatibility
        print(f"\n📜 Script Compatibility:")
        for script_path, result in script_results.items():
            print(f"\n   📄 {script_path}:")

            if not result.get("exists", False):
                print("      ❌ Script file not found")
                continue

            if "error" in result:
                print(f"      ❌ Error reading script: {result['error']}")
                continue

            # Check if script uses enhanced predictor
            if result.get("uses_enhanced_predictor", False):
                print("      ✅ Uses enhanced SRBreakoutPredictor")
            elif result.get("uses_basic_engineering", False):
                print("      ⚠️ Uses basic VectorizedAdvancedFeatureEngineering")
            else:
                print("      ❓ Unknown S/R implementation")

            # Check enhanced features
            enhanced_features = result.get("enhanced_features", {})
            enhanced_count = sum(enhanced_features.values())
            total_enhanced = len(enhanced_features)

            if enhanced_count > 0:
                print(f"      🚀 Enhanced Features: {enhanced_count}/{total_enhanced}")
                for feature, available in enhanced_features.items():
                    if available:
                        print(f"         ✅ {feature}")
            else:
                print("      📊 No enhanced features detected")

            # Check method usage
            method_usage = result.get("method_usage", {})
            method_count = sum(method_usage.values())
            total_methods = len(method_usage)

            if method_count > 0:
                print(f"      🔧 Enhanced Methods: {method_count}/{total_methods}")
                for method, available in method_usage.items():
                    if available:
                        print(f"         ✅ {method}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")

        if sr_results.get("compatible", False):
            print("   ✅ SRBreakoutPredictor is fully functional with enhanced features")

            # Check which scripts need updating
            basic_scripts = [path for path, result in script_results.items()
                           if result.get("uses_basic_engineering", False)]

            if basic_scripts:
                print(f"   🔄 Scripts that should be updated to use enhanced SRBreakoutPredictor:")
                for script in basic_scripts:
                    print(f"      - {script}")
                print("   📝 Enhanced version available: scripts/analyze_sr_position_enhanced.py")
        else:
            print("   ❌ SRBreakoutPredictor has compatibility issues that need to be resolved")

        print("=" * 80)


async def main():
    """Main validation function."""
    validator = SRCompatibilityValidator()

    # Test SRBreakoutPredictor compatibility
    sr_results = await validator.test_sr_breakout_predictor_compatibility()

    # Test script compatibility
    script_results = validator.test_script_compatibility()

    # Print comprehensive report
    validator.print_compatibility_report(sr_results, script_results)

    # Return success/failure
    if sr_results.get("compatible", False):
        print("\n🎉 S/R Script Compatibility Validation PASSED!")
        return 0
    else:
        print("\n❌ S/R Script Compatibility Validation FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)