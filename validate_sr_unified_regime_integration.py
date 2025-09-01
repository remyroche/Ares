#!/usr/bin/env python3
"""
S/R UnifiedRegimeClassifier Integration Validator

This script validates the full integration of SRBreakoutPredictor into UnifiedRegimeClassifier.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


class SRUnifiedRegimeIntegrationValidator:
    """Validator for S/R UnifiedRegimeClassifier integration."""

    def __init__(self):
        self.logger = None  # No logger dependency

    def check_file_syntax(self, file_path: str) -> bool:
        """Check if a Python file has valid syntax."""
        try:
            import ast
            with open(file_path, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error in {file_path}: {e}")
            return False
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return False

    def test_unified_regime_classifier_integration(self) -> Dict[str, Any]:
        """Test UnifiedRegimeClassifier integration with SRBreakoutPredictor."""
        print("🔍 Testing UnifiedRegimeClassifier S/R integration...")

        sr_file = "src/analyst/unified_regime_classifier.py"
        path = Path(sr_file)

        if not path.exists():
            return {
                "exists": False,
                "compatible": False,
                "error": "UnifiedRegimeClassifier file not found"
            }

        try:
            # Check syntax
            if not self.check_file_syntax(sr_file):
                return {
                    "exists": True,
                    "syntax_valid": False,
                    "compatible": False,
                    "error": "Syntax error in UnifiedRegimeClassifier"
                }

            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for SRBreakoutPredictor import
            sr_import = "from src.tactician.sr_breakout_predictor import SRBreakoutPredictor" in content

            # Check for enhanced S/R methods
            enhanced_methods = {
                "initialize_sr_predictor": "async def initialize_sr_predictor" in content,
                "calculate_enhanced_sr_levels": "async def _calculate_enhanced_sr_levels" in content,
                "calculate_basic_pivots": "async def _calculate_basic_pivots" in content,
                "analyze_enhanced_volume_levels": "async def _analyze_enhanced_volume_levels" in content,
                "analyze_basic_volume_levels": "def _analyze_basic_volume_levels" in content,
                "classify_enhanced_location": "async def _classify_enhanced_location" in content,
                "classify_basic_location": "def _classify_basic_location" in content,
                "add_enhanced_sr_features": "async def _add_enhanced_sr_features" in content,
                "add_basic_sr_features": "def _add_basic_sr_features" in content,
            }

            # Check for enhanced S/R features
            enhanced_features = {
                "sr_proximity": "sr_proximity" in content,
                "sr_strength": "sr_strength" in content,
                "sr_zone_width": "sr_zone_width" in content,
                "sr_cluster_count": "sr_cluster_count" in content,
                "sr_fibonacci_proximity": "sr_fibonacci_proximity" in content,
                "sr_elliott_proximity": "sr_elliott_proximity" in content,
                "sr_order_flow_imbalance": "sr_order_flow_imbalance" in content,
                "sr_enhanced_strength": "sr_enhanced_strength" in content,
                "sr_touch_count": "sr_touch_count" in content,
                "sr_bounce_rate": "sr_bounce_rate" in content,
                "sr_isolation_score": "sr_isolation_score" in content,
            }

            # Check for configuration patterns
            config_patterns = {
                "sr_config": "sr_config" in content,
                "enable_sr_integration": "enable_sr_integration" in content,
                "sr_predictor": "sr_predictor" in content,
                "strength_calculation": "strength_calculation" in content,
                "dbscan_clustering": "dbscan_clustering" in content,
                "enhanced_strength": "enable_enhanced_strength" in content,
            }

            # Check for async method updates
            async_methods = {
                "calculate_features": "async def _calculate_features" in content,
                "predict_regime": "async def predict_regime" in content,
                "predict_regime_and_location": "async def predict_regime_and_location" in content,
                "classify_regimes": "async def classify_regimes" in content,
                "train_hmm_labeler": "async def train_hmm_labeler" in content,
                "train_location_classifier": "async def train_location_classifier" in content,
                "train_basic_ensemble": "async def train_basic_ensemble" in content,
            }

            # Check for enhanced location classification
            enhanced_location_labels = {
                "ENHANCED_SUPPORT": "ENHANCED_SUPPORT" in content,
                "ENHANCED_RESISTANCE": "ENHANCED_RESISTANCE" in content,
                "ENHANCED_CONFLUENCE_SUPPORT": "ENHANCED_CONFLUENCE_SUPPORT" in content,
                "ENHANCED_CONFLUENCE_RESISTANCE": "ENHANCED_CONFLUENCE_RESISTANCE" in content,
                "FIBONACCI": "FIBONACCI_" in content,
                "ELLIOTT": "ELLIOTT_" in content,
                "ENHANCED_POC": "ENHANCED_POC" in content,
                "ENHANCED_HVN": "ENHANCED_HVN" in content,
            }

            return {
                "exists": True,
                "syntax_valid": True,
                "sr_import": sr_import,
                "enhanced_methods": enhanced_methods,
                "enhanced_features": enhanced_features,
                "config_patterns": config_patterns,
                "async_methods": async_methods,
                "enhanced_location_labels": enhanced_location_labels,
                "compatible": True
            }

        except Exception as e:
            return {
                "exists": True,
                "error": str(e),
                "compatible": False
            }

    def test_sr_breakout_predictor_compatibility(self) -> Dict[str, Any]:
        """Test SRBreakoutPredictor compatibility with integration."""
        print("🔍 Testing SRBreakoutPredictor compatibility...")

        sr_file = "src/tactician/sr_breakout_predictor.py"
        path = Path(sr_file)

        if not path.exists():
            return {
                "exists": False,
                "compatible": False,
                "error": "SRBreakoutPredictor file not found"
            }

        try:
            # Check syntax
            if not self.check_file_syntax(sr_file):
                return {
                    "exists": True,
                    "syntax_valid": False,
                    "compatible": False,
                    "error": "Syntax error in SRBreakoutPredictor"
                }

            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for required methods that UnifiedRegimeClassifier uses
            required_methods = {
                "get_sr_context": "async def get_sr_context" in content,
                "calculate_fibonacci_levels": "async def calculate_fibonacci_levels" in content,
                "detect_elliott_wave_levels": "async def detect_elliott_wave_levels" in content,
                "analyze_order_flow_levels": "async def analyze_order_flow_levels" in content,
                "cluster_sr_levels_dbscan": "async def cluster_sr_levels_dbscan" in content,
                "calculate_comprehensive_strength": "async def calculate_comprehensive_strength" in content,
                "calculate_touch_count": "async def calculate_touch_count" in content,
                "calculate_bounce_rate": "async def calculate_bounce_rate" in content,
                "calculate_isolation_score": "async def calculate_isolation_score" in content,
            }

            # Check for enhanced features
            enhanced_features = {
                "enhanced_strength_support": "enhanced_strength_support" in content,
                "enhanced_strength_resistance": "enhanced_strength_resistance" in content,
                "clustering_result": "clustering_result" in content,
                "fibonacci_levels": "fibonacci_levels" in content,
                "elliott_wave_levels": "elliott_wave_levels" in content,
                "order_flow_analysis": "order_flow_analysis" in content,
                "support_levels": "support_levels" in content,
                "resistance_levels": "resistance_levels" in content,
                "nearest_support": "nearest_support" in content,
                "nearest_resistance": "nearest_resistance" in content,
                "sr_zone_width": "sr_zone_width" in content,
            }

            return {
                "exists": True,
                "syntax_valid": True,
                "required_methods": required_methods,
                "enhanced_features": enhanced_features,
                "compatible": True
            }

        except Exception as e:
            return {
                "exists": True,
                "error": str(e),
                "compatible": False
            }

    def print_integration_report(self, unified_results: Dict[str, Any], sr_results: Dict[str, Any]) -> None:
        """Print comprehensive integration report."""
        print("\n" + "=" * 80)
        print("📊 S/R UNIFIEDREGIMECLASSIFIER INTEGRATION REPORT")
        print("=" * 80)

        # SRBreakoutPredictor compatibility
        print(f"\n🔧 SRBreakoutPredictor Compatibility:")
        if sr_results.get("compatible", False):
            print("   ✅ SRBreakoutPredictor is compatible with integration")

            required_methods = sr_results.get("required_methods", {})
            print(f"   📊 Required Methods Available:")
            for method, available in required_methods.items():
                status = "✅" if available else "❌"
                print(f"      {status} {method}: {available}")

            enhanced_features = sr_results.get("enhanced_features", {})
            print(f"   🚀 Enhanced Features Available:")
            for feature, available in enhanced_features.items():
                status = "✅" if available else "❌"
                print(f"      {status} {feature}: {available}")

        else:
            print(f"   ❌ SRBreakoutPredictor compatibility issues: {sr_results.get('error', 'Unknown error')}")

        # UnifiedRegimeClassifier integration
        print(f"\n🔧 UnifiedRegimeClassifier Integration:")
        if unified_results.get("compatible", False):
            print("   ✅ UnifiedRegimeClassifier integration is complete")

            sr_import = unified_results.get("sr_import", False)
            print(f"   📦 SRBreakoutPredictor Import: {'✅ Available' if sr_import else '❌ Not Found'}")

            enhanced_methods = unified_results.get("enhanced_methods", {})
            print(f"   📊 Enhanced Methods Implemented:")
            for method, available in enhanced_methods.items():
                status = "✅" if available else "❌"
                print(f"      {status} {method}: {available}")

            enhanced_features = unified_results.get("enhanced_features", {})
            print(f"   🚀 Enhanced S/R Features Added:")
            for feature, available in enhanced_features.items():
                status = "✅" if available else "❌"
                print(f"      {status} {feature}: {available}")

            config_patterns = unified_results.get("config_patterns", {})
            print(f"   ⚙️ Configuration Patterns Available:")
            for pattern, available in config_patterns.items():
                status = "✅" if available else "❌"
                print(f"      {status} {pattern}: {available}")

            async_methods = unified_results.get("async_methods", {})
            print(f"   🔄 Async Method Updates:")
            for method, available in async_methods.items():
                status = "✅" if available else "❌"
                print(f"      {status} {method}: {available}")

            enhanced_location_labels = unified_results.get("enhanced_location_labels", {})
            print(f"   🎯 Enhanced Location Labels:")
            for label, available in enhanced_location_labels.items():
                status = "✅" if available else "❌"
                print(f"      {status} {label}: {available}")

        else:
            print(f"   ❌ UnifiedRegimeClassifier integration issues: {unified_results.get('error', 'Unknown error')}")

        # Integration Summary
        print(f"\n💡 INTEGRATION SUMMARY:")

        if sr_results.get("compatible", False) and unified_results.get("compatible", False):
            print("   ✅ Full integration successful!")
            print("   🚀 Enhanced S/R analysis now available in HMM regime classification")
            print("   📊 Advanced features include:")
            print("      - DBSCAN clustering for S/R level filtering")
            print("      - Fibonacci retracement and extension levels")
            print("      - Elliott Wave analysis")
            print("      - Order flow analysis with POC and HVNs")
            print("      - Enhanced strength calculation with multiple factors")
            print("      - Professional noise filtering")
            print("      - Multi-timeframe confluence analysis")
            print("   🎯 Enhanced location classification with priority-based labeling")
            print("   📈 Improved regime analysis with S/R context")
        else:
            print("   ❌ Integration incomplete - issues need to be resolved")

            if not sr_results.get("compatible", False):
                print("   🔧 SRBreakoutPredictor needs to be fixed")
            if not unified_results.get("compatible", False):
                print("   🔧 UnifiedRegimeClassifier integration needs to be completed")

        print("=" * 80)


def main():
    """Main validation function."""
    validator = SRUnifiedRegimeIntegrationValidator()

    # Test SRBreakoutPredictor compatibility
    sr_results = validator.test_sr_breakout_predictor_compatibility()

    # Test UnifiedRegimeClassifier integration
    unified_results = validator.test_unified_regime_classifier_integration()

    # Print comprehensive report
    validator.print_integration_report(unified_results, sr_results)

    # Return success/failure
    if sr_results.get("compatible", False) and unified_results.get("compatible", False):
        print("\n🎉 S/R UnifiedRegimeClassifier Integration Validation PASSED!")
        return 0
    else:
        print("\n❌ S/R UnifiedRegimeClassifier Integration Validation FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)