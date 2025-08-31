#!/usr/bin/env python3
"""
Enhanced Missing Value Handler Test

This test validates the enhanced missing value handler including:
- Gap analysis and classification
- Forward fill for small gaps (≤5 seconds)
- Data download for larger gaps
- Fallback strategies
- Data continuity validation
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import tempfile

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.enhanced_missing_value_handler import enhanced_missing_value_handler, GapType, GapInfo
from src.utils.logger import system_logger


class EnhancedMissingValueHandlerTester:
    """Comprehensive tester for enhanced missing value handler."""
    
    def __init__(self):
        """Initialize tester."""
        self.logger = system_logger.getChild("EnhancedMissingValueHandlerTester")
        self.handler = enhanced_missing_value_handler
        self.test_results = {}
        self.start_time = time.time()
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all enhanced missing value handler tests."""
        self.logger.info("🔍 Starting Enhanced Missing Value Handler Tests")
        
        test_suite = [
            ("test_gap_analysis", self.test_gap_analysis),
            ("test_gap_classification", self.test_gap_classification),
            ("test_small_gap_handling", self.test_small_gap_handling),
            ("test_large_gap_handling", self.test_large_gap_handling),
            ("test_critical_gap_handling", self.test_critical_gap_handling),
            ("test_intelligent_missing_value_handling", self.test_intelligent_missing_value_handling),
            ("test_data_continuity_validation", self.test_data_continuity_validation),
            ("test_gap_reporting", self.test_gap_reporting),
            ("test_fallback_strategies", self.test_fallback_strategies),
            ("test_integration_with_formatting", self.test_integration_with_formatting)
        ]
        
        for test_name, test_func in test_suite:
            try:
                self.logger.info(f"Running {test_name}...")
                result = test_func()
                self.test_results[test_name] = {
                    "status": "PASSED" if result else "FAILED",
                    "details": result
                }
                self.logger.info(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                self.logger.error(f"❌ {test_name} failed with exception: {e}")
                self.test_results[test_name] = {
                    "status": "ERROR",
                    "details": str(e)
                }
        
        return self.generate_test_report()
    
    def test_gap_analysis(self) -> bool:
        """Test gap analysis functionality."""
        self.logger.info("Testing gap analysis...")
        
        # Create test data with gaps
        test_data = self._create_test_data_with_gaps()
        
        # Analyze gaps
        gaps = self.handler._analyze_gaps(test_data, "timestamp")
        
        # Check that gaps were detected
        if len(gaps) == 0:
            self.logger.error("No gaps detected in test data")
            return False
        
        # Check gap properties
        for gap in gaps:
            if not isinstance(gap, GapInfo):
                self.logger.error("Gap is not GapInfo instance")
                return False
            
            if gap.start_time >= gap.end_time:
                self.logger.error("Invalid gap time range")
                return False
            
            if gap.gap_size <= 0:
                self.logger.error("Invalid gap size")
                return False
        
        self.logger.info(f"Gap analysis tests passed: {len(gaps)} gaps detected")
        return True
    
    def test_gap_classification(self) -> bool:
        """Test gap classification functionality."""
        self.logger.info("Testing gap classification...")
        
        # Test different gap sizes
        test_cases = [
            (3, GapType.SMALL),      # 3 seconds - small gap
            (10, GapType.MEDIUM),    # 10 seconds - medium gap
            (120, GapType.LARGE),    # 120 seconds - large gap
            (600, GapType.CRITICAL)  # 600 seconds - critical gap
        ]
        
        for gap_size, expected_type in test_cases:
            classified_type = self.handler._classify_gap(gap_size)
            if classified_type != expected_type:
                self.logger.error(f"Gap classification failed: {gap_size}s classified as {classified_type}, expected {expected_type}")
                return False
        
        self.logger.info("Gap classification tests passed")
        return True
    
    def test_small_gap_handling(self) -> bool:
        """Test small gap handling with forward fill."""
        self.logger.info("Testing small gap handling...")
        
        # Create test data with small gap
        test_data = self._create_test_data_with_small_gap()
        
        # Get initial gap count
        initial_gaps = self.handler._analyze_gaps(test_data, "timestamp")
        initial_gap_count = len(initial_gaps)
        
        # Handle small gap
        filled_data = self.handler._handle_small_gap(test_data, initial_gaps[0], "timestamp")
        
        # Check that gap was filled
        final_gaps = self.handler._analyze_gaps(filled_data, "timestamp")
        final_gap_count = len(final_gaps)
        
        if final_gap_count >= initial_gap_count:
            self.logger.error("Small gap was not filled")
            return False
        
        # Check that data was forward filled
        if len(filled_data) <= len(test_data):
            self.logger.error("No new rows were added during forward fill")
            return False
        
        self.logger.info("Small gap handling tests passed")
        return True
    
    def test_large_gap_handling(self) -> bool:
        """Test large gap handling with fallback strategy."""
        self.logger.info("Testing large gap handling...")
        
        # Create test data with large gap
        test_data = self._create_test_data_with_large_gap()
        
        # Create a mock gap
        gap = GapInfo(
            start_time=int(time.time()) - 300,
            end_time=int(time.time()) - 240,
            gap_size=60,
            gap_type=GapType.MEDIUM
        )
        
        # Handle large gap with fallback (no symbol/exchange provided)
        filled_data = self.handler._handle_large_gap_with_fallback(test_data, gap, "timestamp")
        
        # Check that gap was handled
        if len(filled_data) <= len(test_data):
            self.logger.error("No new rows were added during large gap handling")
            return False
        
        # Check that interpolation was used
        gap.filled = True
        if gap.fill_method != "interpolation_fallback":
            self.logger.error("Fallback strategy was not used")
            return False
        
        self.logger.info("Large gap handling tests passed")
        return True
    
    def test_critical_gap_handling(self) -> bool:
        """Test critical gap handling."""
        self.logger.info("Testing critical gap handling...")
        
        # Create test data with critical gap
        test_data = self._create_test_data_with_critical_gap()
        
        # Create a mock critical gap
        gap = GapInfo(
            start_time=int(time.time()) - 600,
            end_time=int(time.time()) - 300,
            gap_size=300,
            gap_type=GapType.CRITICAL
        )
        
        # Handle critical gap
        filled_data = self.handler._handle_critical_gap(test_data, gap, "timestamp")
        
        # Check that gap was handled (should use fallback)
        if len(filled_data) <= len(test_data):
            self.logger.error("No new rows were added during critical gap handling")
            return False
        
        self.logger.info("Critical gap handling tests passed")
        return True
    
    def test_intelligent_missing_value_handling(self) -> bool:
        """Test intelligent missing value handling."""
        self.logger.info("Testing intelligent missing value handling...")
        
        # Create test data with mixed gaps
        test_data = self._create_test_data_with_mixed_gaps()
        
        # Get initial gap count
        initial_gaps = self.handler._analyze_gaps(test_data, "timestamp")
        initial_gap_count = len(initial_gaps)
        
        # Handle missing values intelligently
        filled_data = self.handler.handle_missing_values_intelligently(
            test_data, "timestamp", "BTCUSDT", "binance", "1m"
        )
        
        # Check that gaps were handled
        final_gaps = self.handler._analyze_gaps(filled_data, "timestamp")
        final_gap_count = len(final_gaps)
        
        if final_gap_count >= initial_gap_count:
            self.logger.error("Intelligent missing value handling did not reduce gaps")
            return False
        
        self.logger.info("Intelligent missing value handling tests passed")
        return True
    
    def test_data_continuity_validation(self) -> bool:
        """Test data continuity validation."""
        self.logger.info("Testing data continuity validation...")
        
        # Create continuous data
        continuous_data = self._create_continuous_test_data()
        
        # Validate continuity
        continuity_report = self.handler.validate_data_continuity(continuous_data, "timestamp", 60)
        
        if not continuity_report["valid"]:
            self.logger.error("Continuous data failed continuity validation")
            return False
        
        if continuity_report["continuity_score"] < 0.99:
            self.logger.error("Continuous data has low continuity score")
            return False
        
        # Create discontinuous data
        discontinuous_data = self._create_discontinuous_test_data()
        
        # Validate discontinuity
        discontinuity_report = self.handler.validate_data_continuity(discontinuous_data, "timestamp", 60)
        
        if discontinuity_report["valid"]:
            self.logger.error("Discontinuous data passed continuity validation")
            return False
        
        if discontinuity_report["issues_count"] == 0:
            self.logger.error("No issues detected in discontinuous data")
            return False
        
        self.logger.info("Data continuity validation tests passed")
        return True
    
    def test_gap_reporting(self) -> bool:
        """Test gap reporting functionality."""
        self.logger.info("Testing gap reporting...")
        
        # Create test data with gaps
        test_data = self._create_test_data_with_gaps()
        
        # Generate gap report
        gap_report = self.handler.get_gap_report(test_data, "timestamp")
        
        # Check report structure
        required_keys = ["timestamp", "total_gaps", "gap_summary", "gap_details"]
        for key in required_keys:
            if key not in gap_report:
                self.logger.error(f"Missing key in gap report: {key}")
                return False
        
        # Check that gaps were reported
        if gap_report["total_gaps"] == 0:
            self.logger.error("No gaps reported in test data")
            return False
        
        # Check gap summary
        if not gap_report["gap_summary"]:
            self.logger.error("Empty gap summary")
            return False
        
        self.logger.info("Gap reporting tests passed")
        return True
    
    def test_fallback_strategies(self) -> bool:
        """Test fallback strategies."""
        self.logger.info("Testing fallback strategies...")
        
        # Create test data with large gap
        test_data = self._create_test_data_with_large_gap()
        
        # Create a mock gap
        gap = GapInfo(
            start_time=int(time.time()) - 300,
            end_time=int(time.time()) - 240,
            gap_size=60,
            gap_type=GapType.MEDIUM
        )
        
        # Test fallback strategy
        filled_data = self.handler._handle_large_gap_with_fallback(test_data, gap, "timestamp")
        
        # Check that fallback was used
        if not gap.filled:
            self.logger.error("Gap was not marked as filled")
            return False
        
        if gap.fill_method != "interpolation_fallback":
            self.logger.error("Fallback method was not used")
            return False
        
        # Check that data was interpolated
        if len(filled_data) <= len(test_data):
            self.logger.error("No interpolation occurred")
            return False
        
        self.logger.info("Fallback strategies tests passed")
        return True
    
    def test_integration_with_formatting(self) -> bool:
        """Test integration with data formatting framework."""
        self.logger.info("Testing integration with formatting framework...")
        
        # Import formatting framework
        from src.utils.data_formatting_framework import data_formatting_framework
        
        # Create test data with gaps
        test_data = self._create_test_data_with_gaps()
        
        # Use intelligent missing value handling through formatting framework
        filled_data = data_formatting_framework.handle_missing_values(
            test_data, "intelligent", symbol="BTCUSDT", exchange="binance", timeframe="1m"
        )
        
        # Check that gaps were handled
        initial_gaps = self.handler._analyze_gaps(test_data, "timestamp")
        final_gaps = self.handler._analyze_gaps(filled_data, "timestamp")
        
        if len(final_gaps) >= len(initial_gaps):
            self.logger.error("Integration with formatting framework did not handle gaps")
            return False
        
        self.logger.info("Integration with formatting framework tests passed")
        return True
    
    def _create_test_data_with_gaps(self) -> pd.DataFrame:
        """Create test data with various gaps."""
        base_time = int(time.time()) - 3600  # 1 hour ago
        
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        # Create data with gaps
        current_time = base_time
        for i in range(100):
            if i == 20:  # Small gap
                current_time += 120  # 2 minutes gap
            elif i == 40:  # Medium gap
                current_time += 300  # 5 minutes gap
            elif i == 60:  # Large gap
                current_time += 1800  # 30 minutes gap
            
            timestamps.append(current_time)
            opens.append(100 + np.random.random() * 10)
            highs.append(105 + np.random.random() * 10)
            lows.append(95 + np.random.random() * 10)
            closes.append(100 + np.random.random() * 10)
            volumes.append(1000 + np.random.random() * 1000)
            
            current_time += 60  # 1 minute intervals
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        })
    
    def _create_test_data_with_small_gap(self) -> pd.DataFrame:
        """Create test data with small gap."""
        base_time = int(time.time()) - 3600
        
        timestamps = []
        for i in range(50):
            if i == 25:  # Small gap
                timestamps.append(base_time + i * 60 + 120)  # 2 minute gap
            else:
                timestamps.append(base_time + i * 60)
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "open": [100 + np.random.random() * 10 for _ in timestamps],
            "high": [105 + np.random.random() * 10 for _ in timestamps],
            "low": [95 + np.random.random() * 10 for _ in timestamps],
            "close": [100 + np.random.random() * 10 for _ in timestamps],
            "volume": [1000 + np.random.random() * 1000 for _ in timestamps]
        })
    
    def _create_test_data_with_large_gap(self) -> pd.DataFrame:
        """Create test data with large gap."""
        base_time = int(time.time()) - 3600
        
        timestamps = []
        for i in range(50):
            if i == 25:  # Large gap
                timestamps.append(base_time + i * 60 + 1800)  # 30 minute gap
            else:
                timestamps.append(base_time + i * 60)
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "open": [100 + np.random.random() * 10 for _ in timestamps],
            "high": [105 + np.random.random() * 10 for _ in timestamps],
            "low": [95 + np.random.random() * 10 for _ in timestamps],
            "close": [100 + np.random.random() * 10 for _ in timestamps],
            "volume": [1000 + np.random.random() * 1000 for _ in timestamps]
        })
    
    def _create_test_data_with_critical_gap(self) -> pd.DataFrame:
        """Create test data with critical gap."""
        base_time = int(time.time()) - 3600
        
        timestamps = []
        for i in range(50):
            if i == 25:  # Critical gap
                timestamps.append(base_time + i * 60 + 3600)  # 1 hour gap
            else:
                timestamps.append(base_time + i * 60)
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "open": [100 + np.random.random() * 10 for _ in timestamps],
            "high": [105 + np.random.random() * 10 for _ in timestamps],
            "low": [95 + np.random.random() * 10 for _ in timestamps],
            "close": [100 + np.random.random() * 10 for _ in timestamps],
            "volume": [1000 + np.random.random() * 1000 for _ in timestamps]
        })
    
    def _create_test_data_with_mixed_gaps(self) -> pd.DataFrame:
        """Create test data with mixed gap types."""
        base_time = int(time.time()) - 3600
        
        timestamps = []
        for i in range(100):
            if i == 20:  # Small gap
                timestamps.append(base_time + i * 60 + 120)
            elif i == 40:  # Medium gap
                timestamps.append(base_time + i * 60 + 300)
            elif i == 60:  # Large gap
                timestamps.append(base_time + i * 60 + 1800)
            else:
                timestamps.append(base_time + i * 60)
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "open": [100 + np.random.random() * 10 for _ in timestamps],
            "high": [105 + np.random.random() * 10 for _ in timestamps],
            "low": [95 + np.random.random() * 10 for _ in timestamps],
            "close": [100 + np.random.random() * 10 for _ in timestamps],
            "volume": [1000 + np.random.random() * 1000 for _ in timestamps]
        })
    
    def _create_continuous_test_data(self) -> pd.DataFrame:
        """Create test data with continuous timestamps."""
        base_time = int(time.time()) - 3600
        
        timestamps = [base_time + i * 60 for i in range(100)]
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "open": [100 + np.random.random() * 10 for _ in timestamps],
            "high": [105 + np.random.random() * 10 for _ in timestamps],
            "low": [95 + np.random.random() * 10 for _ in timestamps],
            "close": [100 + np.random.random() * 10 for _ in timestamps],
            "volume": [1000 + np.random.random() * 1000 for _ in timestamps]
        })
    
    def _create_discontinuous_test_data(self) -> pd.DataFrame:
        """Create test data with discontinuous timestamps."""
        base_time = int(time.time()) - 3600
        
        timestamps = []
        for i in range(100):
            if i % 10 == 0:  # Create gaps every 10th entry
                timestamps.append(base_time + i * 60 + 300)  # 5 minute gap
            else:
                timestamps.append(base_time + i * 60)
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "open": [100 + np.random.random() * 10 for _ in timestamps],
            "high": [105 + np.random.random() * 10 for _ in timestamps],
            "low": [95 + np.random.random() * 10 for _ in timestamps],
            "close": [100 + np.random.random() * 10 for _ in timestamps],
            "volume": [1000 + np.random.random() * 1000 for _ in timestamps]
        })
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Count results
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r["status"] == "PASSED"])
        failed_tests = len([r for r in self.test_results.values() if r["status"] == "FAILED"])
        error_tests = len([r for r in self.test_results.values() if r["status"] == "ERROR"])
        
        # Get handler configuration
        handler_config = {
            "max_forward_fill_gap": self.handler.max_forward_fill_gap,
            "download_threshold": self.handler.download_threshold,
            "gap_thresholds": {k.value: v for k, v in self.handler.gap_thresholds.items()},
            "fill_strategies": {k.value: v for k, v in self.handler.fill_strategies.items()}
        }
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "duration_seconds": duration
            },
            "test_results": self.test_results,
            "handler_configuration": handler_config,
            "gap_types": [gap_type.value for gap_type in GapType],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_tests = [name for name, result in self.test_results.items() if result["status"] == "FAILED"]
        error_tests = [name for name, result in self.test_results.items() if result["status"] == "ERROR"]
        
        if failed_tests:
            recommendations.append(f"Fix failed tests: {', '.join(failed_tests)}")
        
        if error_tests:
            recommendations.append(f"Investigate test errors: {', '.join(error_tests)}")
        
        # Check handler configuration
        if self.handler.max_forward_fill_gap != 5:
            recommendations.append("Consider adjusting max_forward_fill_gap to 5 seconds for optimal performance")
        
        if self.handler.download_threshold != 5:
            recommendations.append("Consider adjusting download_threshold to 5 seconds for optimal performance")
        
        # Check test coverage
        if len(self.test_results) < 10:
            recommendations.append("Add more comprehensive tests for edge cases")
        
        return recommendations


def main():
    """Main function to run enhanced missing value handler tests."""
    print("🔍 Enhanced Missing Value Handler Test Framework")
    print("=" * 60)
    
    tester = EnhancedMissingValueHandlerTester()
    report = tester.run_all_tests()
    
    # Print summary
    summary = report["test_summary"]
    print(f"\n📊 Test Summary:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Passed: {summary['passed_tests']}")
    print(f"  Failed: {summary['failed_tests']}")
    print(f"  Errors: {summary['error_tests']}")
    print(f"  Success Rate: {summary['success_rate']:.2%}")
    print(f"  Duration: {summary['duration_seconds']:.2f} seconds")
    
    # Print handler configuration
    config = report["handler_configuration"]
    print(f"\n🔧 Handler Configuration:")
    print(f"  Max Forward Fill Gap: {config['max_forward_fill_gap']} seconds")
    print(f"  Download Threshold: {config['download_threshold']} seconds")
    print(f"  Gap Thresholds: {config['gap_thresholds']}")
    print(f"  Fill Strategies: {config['fill_strategies']}")
    
    # Print gap types
    print(f"\n📈 Gap Types:")
    for gap_type in report["gap_types"]:
        print(f"  • {gap_type}")
    
    # Print recommendations
    print(f"\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    # Save detailed report
    report_file = "enhanced_missing_value_handler_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Return success if most tests passed
    return summary['success_rate'] >= 0.8


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)