"""
Aggtrades Data Quality Verification - Example Usage

This module demonstrates how to use the comprehensive aggtrades data quality verification system
for regular data quality checks in different pipeline steps.

Key Features Demonstrated:
- Basic quality verification
- Custom configuration
- Auto-fix capabilities
- Integration with existing pipelines
- Regular monitoring setup
- Batch processing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

from .aggtrades_quality_verification import (
    AggtradesQualityVerifier,
    verify_aggtrades_quality,
    create_aggtrades_quality_config,
    QualityIssueSeverity
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_aggtrades_data(rows: int = 1000, with_issues: bool = True) -> pd.DataFrame:
    """Create sample aggtrades data for testing."""
    logger.info(f"Creating sample aggtrades data with {rows} rows")
    
    # Base timestamp
    start_time = datetime.now() - timedelta(hours=1)
    timestamps = [start_time + timedelta(milliseconds=i*100) for i in range(rows)]
    
    # Base price data
    base_price = 50000.0
    prices = [base_price + np.random.normal(0, 100) for _ in range(rows)]
    
    # Base volume data
    base_volume = 1.0
    volumes = [base_volume + np.random.exponential(0.5) for _ in range(rows)]
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'quantity': volumes,
        'first_trade_id': range(1000, 1000 + rows),
        'last_trade_id': range(1000, 1000 + rows),
        'is_buyer_maker': np.random.choice([True, False], rows)
    })
    
    if with_issues:
        # Introduce some quality issues for testing
        logger.info("Introducing quality issues for testing")
        
        # 1. Add some timestamp gaps
        gap_indices = np.random.choice(rows, size=5, replace=False)
        for idx in gap_indices:
            data.loc[idx, 'timestamp'] += timedelta(seconds=2)  # 2-second gap
        
        # 2. Add some duplicates
        duplicate_indices = np.random.choice(rows, size=3, replace=False)
        for idx in duplicate_indices:
            data.loc[idx] = data.iloc[0]  # Copy first row
        
        # 3. Add some negative prices
        negative_price_indices = np.random.choice(rows, size=2, replace=False)
        for idx in negative_price_indices:
            data.loc[idx, 'price'] = -100.0
        
        # 4. Add some negative volumes
        negative_volume_indices = np.random.choice(rows, size=2, replace=False)
        for idx in negative_volume_indices:
            data.loc[idx, 'quantity'] = -0.5
        
        # 5. Add some price outliers
        outlier_indices = np.random.choice(rows, size=3, replace=False)
        for idx in outlier_indices:
            data.loc[idx, 'price'] = base_price * 10  # 10x normal price
    
    return data


def example_basic_verification():
    """Example 1: Basic quality verification."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: Basic Quality Verification")
    logger.info("=" * 60)
    
    # Create sample data with issues
    data = create_sample_aggtrades_data(1000, with_issues=True)
    logger.info(f"Created sample data with {len(data)} rows")
    
    # Basic verification
    cleaned_data, report = verify_aggtrades_quality(data)
    
    # Display results
    logger.info(f"Original rows: {len(data)}")
    logger.info(f"Cleaned rows: {len(cleaned_data)}")
    logger.info(f"Quality score: {report.quality_score:.3f}")
    logger.info(f"Issues found: {len(report.issues)}")
    
    # Show issue summary
    for issue in report.issues:
        logger.info(f"  - {issue.severity.value.upper()}: {issue.message}")
    
    return cleaned_data, report


def example_custom_configuration():
    """Example 2: Custom configuration."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: Custom Configuration")
    logger.info("=" * 60)
    
    # Create custom configuration
    config = create_aggtrades_quality_config(
        max_timestamp_gap_seconds=1.0,  # More lenient gap threshold
        max_duplicate_ratio=0.005,      # More lenient duplicate threshold
        price_outlier_threshold=3.0,    # More sensitive outlier detection
        duplicate_action="remove",      # Auto-remove duplicates
        price_negative_action="remove"  # Auto-remove negative prices
    )
    
    logger.info("Custom configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create sample data
    data = create_sample_aggtrades_data(500, with_issues=True)
    
    # Verify with custom config
    verifier = AggtradesQualityVerifier(config, logger)
    cleaned_data, report = verifier.verify_aggtrades_quality(data, auto_fix=True)
    
    logger.info(f"Quality score with custom config: {report.quality_score:.3f}")
    logger.info(f"Issues found: {len(report.issues)}")
    
    return cleaned_data, report


def example_auto_fix():
    """Example 3: Auto-fix capabilities."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: Auto-Fix Capabilities")
    logger.info("=" * 60)
    
    # Create data with issues
    data = create_sample_aggtrades_data(800, with_issues=True)
    original_rows = len(data)
    
    # Configuration for auto-fix
    config = create_aggtrades_quality_config(
        duplicate_action="remove",
        price_negative_action="remove",
        volume_negative_action="remove"
    )
    
    # Verify with auto-fix
    verifier = AggtradesQualityVerifier(config, logger)
    cleaned_data, report = verifier.verify_aggtrades_quality(data, auto_fix=True)
    
    logger.info(f"Original rows: {original_rows}")
    logger.info(f"Cleaned rows: {len(cleaned_data)}")
    logger.info(f"Rows removed: {original_rows - len(cleaned_data)}")
    logger.info(f"Quality score: {report.quality_score:.3f}")
    
    # Show what was fixed
    for issue in report.issues:
        if issue.action.value == "remove":
            logger.info(f"  Auto-fixed: {issue.message}")
    
    return cleaned_data, report


def example_step_integration():
    """Example 4: Integration with pipeline steps."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 4: Pipeline Step Integration")
    logger.info("=" * 60)
    
    # Simulate different pipeline steps
    steps = [
        {"name": "data_loading", "description": "After loading raw data"},
        {"name": "data_preprocessing", "description": "After preprocessing"},
        {"name": "feature_engineering", "description": "After feature engineering"},
        {"name": "model_training", "description": "Before model training"}
    ]
    
    # Create sample data for each step
    data = create_sample_aggtrades_data(600, with_issues=True)
    
    # Quality verification configuration
    config = create_aggtrades_quality_config(
        max_timestamp_gap_seconds=0.5,
        max_duplicate_ratio=0.001,
        duplicate_action="remove"
    )
    
    verifier = AggtradesQualityVerifier(config, logger)
    
    for step in steps:
        logger.info(f"\n--- {step['name'].upper()}: {step['description']} ---")
        
        # Verify quality at this step
        cleaned_data, report = verifier.verify_aggtrades_quality(data, auto_fix=True)
        
        logger.info(f"Quality score: {report.quality_score:.3f}")
        logger.info(f"Issues found: {len(report.issues)}")
        
        # Export report for this step
        report_filename = f"quality_report_{step['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        verifier.export_quality_report(report, report_filename)
        logger.info(f"Report exported: {report_filename}")
        
        # Update data for next step (simulate processing)
        data = cleaned_data.copy()
    
    return data


def example_monitoring_setup():
    """Example 5: Regular monitoring setup."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 5: Regular Monitoring Setup")
    logger.info("=" * 60)
    
    # Configuration for monitoring
    monitoring_config = create_aggtrades_quality_config(
        max_timestamp_gap_seconds=0.5,
        max_duplicate_ratio=0.001,
        price_outlier_threshold=4.0,
        volume_outlier_threshold=4.0,
        # Actions for monitoring
        timestamp_gap_action="warn",
        duplicate_action="remove",
        price_negative_action="fail",
        volume_negative_action="fail",
        price_outlier_action="warn",
        volume_outlier_action="warn"
    )
    
    def monitor_data_quality(data: pd.DataFrame, step_name: str) -> Dict[str, any]:
        """Monitor data quality for a specific step."""
        logger.info(f"🔍 Monitoring data quality for step: {step_name}")
        
        verifier = AggtradesQualityVerifier(monitoring_config, logger)
        cleaned_data, report = verifier.verify_aggtrades_quality(data, auto_fix=True)
        
        # Check for critical issues
        critical_issues = [issue for issue in report.issues if issue.severity == QualityIssueSeverity.CRITICAL]
        error_issues = [issue for issue in report.issues if issue.severity == QualityIssueSeverity.ERROR]
        
        monitoring_result = {
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "quality_score": report.quality_score,
            "total_issues": len(report.issues),
            "critical_issues": len(critical_issues),
            "error_issues": len(error_issues),
            "rows_processed": len(data),
            "rows_after_cleaning": len(cleaned_data),
            "status": "PASS" if len(critical_issues) == 0 and len(error_issues) == 0 else "FAIL"
        }
        
        # Log monitoring result
        if monitoring_result["status"] == "PASS":
            logger.info(f"✅ {step_name}: Quality check PASSED (score: {report.quality_score:.3f})")
        else:
            logger.error(f"❌ {step_name}: Quality check FAILED (score: {report.quality_score:.3f})")
            logger.error(f"   Critical issues: {len(critical_issues)}")
            logger.error(f"   Error issues: {len(error_issues)}")
        
        return monitoring_result
    
    # Simulate monitoring across multiple steps
    data = create_sample_aggtrades_data(400, with_issues=True)
    
    monitoring_results = []
    steps = ["data_loading", "preprocessing", "feature_engineering", "model_training"]
    
    for step in steps:
        result = monitor_data_quality(data, step)
        monitoring_results.append(result)
        
        # Simulate some processing that might introduce issues
        if step == "preprocessing":
            # Simulate adding some issues during preprocessing
            data.loc[0, 'price'] = -50.0  # Add negative price
    
    # Summary
    logger.info("\n📊 Monitoring Summary:")
    for result in monitoring_results:
        status_emoji = "✅" if result["status"] == "PASS" else "❌"
        logger.info(f"  {status_emoji} {result['step_name']}: {result['status']} (score: {result['quality_score']:.3f})")
    
    return monitoring_results


def example_batch_processing():
    """Example 6: Batch processing multiple datasets."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 6: Batch Processing")
    logger.info("=" * 60)
    
    # Create multiple datasets
    datasets = {
        "BTCUSDT_1m": create_sample_aggtrades_data(1000, with_issues=True),
        "ETHUSDT_1m": create_sample_aggtrades_data(800, with_issues=True),
        "ADAUSDT_1m": create_sample_aggtrades_data(600, with_issues=True)
    }
    
    # Batch processing configuration
    batch_config = create_aggtrades_quality_config(
        max_timestamp_gap_seconds=0.5,
        max_duplicate_ratio=0.001,
        duplicate_action="remove",
        price_negative_action="remove",
        volume_negative_action="remove"
    )
    
    verifier = AggtradesQualityVerifier(batch_config, logger)
    
    batch_results = {}
    
    for symbol, data in datasets.items():
        logger.info(f"\n🔄 Processing {symbol}...")
        
        try:
            cleaned_data, report = verifier.verify_aggtrades_quality(data, auto_fix=True)
            
            batch_results[symbol] = {
                "status": "SUCCESS",
                "original_rows": len(data),
                "cleaned_rows": len(cleaned_data),
                "quality_score": report.quality_score,
                "issues_found": len(report.issues),
                "cleaned_data": cleaned_data
            }
            
            logger.info(f"  ✅ {symbol}: {len(data)} → {len(cleaned_data)} rows (score: {report.quality_score:.3f})")
            
        except Exception as e:
            batch_results[symbol] = {
                "status": "FAILED",
                "error": str(e),
                "original_rows": len(data)
            }
            logger.error(f"  ❌ {symbol}: Failed - {e}")
    
    # Batch summary
    logger.info("\n📊 Batch Processing Summary:")
    successful = sum(1 for result in batch_results.values() if result["status"] == "SUCCESS")
    total = len(batch_results)
    logger.info(f"  Successful: {successful}/{total}")
    
    for symbol, result in batch_results.items():
        if result["status"] == "SUCCESS":
            logger.info(f"  ✅ {symbol}: {result['original_rows']} → {result['cleaned_rows']} rows")
        else:
            logger.info(f"  ❌ {symbol}: FAILED")
    
    return batch_results


def main():
    """Run all examples."""
    logger.info("🚀 Starting Aggtrades Data Quality Verification Examples")
    logger.info("=" * 80)
    
    try:
        # Run all examples
        example_basic_verification()
        example_custom_configuration()
        example_auto_fix()
        example_step_integration()
        example_monitoring_setup()
        example_batch_processing()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ All examples completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Example execution failed: {e}")
        raise


if __name__ == "__main__":
    main()