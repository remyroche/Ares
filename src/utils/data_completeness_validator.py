"""
Data Completeness Validator

This module provides functionality to validate the completeness of existing data
from step1 and step1_5 without triggering new downloads. It checks for data gaps
and provides warnings for incomplete data.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

# Optional pandas import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataCompletenessValidator:
    """Validates completeness of existing data from step1 and step1_5."""
    
    def __init__(self, data_cache_dir: str = "data_cache"):
        self.data_cache_dir = Path(data_cache_dir)
        self.logger = logging.getLogger(f"{__name__}.DataCompletenessValidator")
    
    def validate_step1_data_completeness(
        self, 
        symbol: str, 
        exchange: str,
        expected_lookback_days: int = 60
    ) -> Dict[str, Any]:
        """
        Validate completeness of step1 data collection.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'BINANCE')
            expected_lookback_days: Expected number of days of data
            
        Returns:
            Dict with validation results including warnings and completeness status
        """
        self.logger.info(f"🔍 Validating step1 data completeness for {symbol} on {exchange}")
        
        validation_result = {
            "step1_complete": False,
            "step1_5_complete": False,
            "warnings": [],
            "data_files": {},
            "gaps": [],
            "recommendations": []
        }
        
        # Check for step1 data files
        step1_files = self._find_step1_files(symbol, exchange)
        validation_result["data_files"]["step1"] = step1_files
        
        if not step1_files:
            validation_result["warnings"].append(
                f"No step1 data files found for {symbol} on {exchange}"
            )
            validation_result["recommendations"].append(
                "Run step1_data_collection first: python ares_launcher.py full --symbol {symbol} --exchange {exchange} --step step1_data_collection"
            )
            return validation_result
        
        # Check for step1_5 data files
        step1_5_files = self._find_step1_5_files(symbol, exchange)
        validation_result["data_files"]["step1_5"] = step1_5_files
        
        if not step1_5_files:
            validation_result["warnings"].append(
                f"No step1_5 data files found for {symbol} on {exchange}"
            )
            validation_result["recommendations"].append(
                "Run step1_5_data_converter first: python ares_launcher.py full --symbol {symbol} --exchange {exchange} --step step1_5_data_converter"
            )
            return validation_result
        
        # Validate data completeness for each file type
        step1_complete = self._validate_step1_completeness(step1_files, expected_lookback_days)
        step1_5_complete = self._validate_step1_5_completeness(step1_5_files, expected_lookback_days)
        
        validation_result["step1_complete"] = step1_complete
        validation_result["step1_5_complete"] = step1_5_complete
        
        # Check for data gaps
        gaps = self._detect_data_gaps(step1_files, step1_5_files, expected_lookback_days)
        validation_result["gaps"] = gaps
        
        if gaps:
            validation_result["warnings"].extend([
                f"Data gaps detected: {', '.join(gaps)}"
            ])
            validation_result["recommendations"].append(
                "Consider running data collection to fill gaps, but proceeding with existing data"
            )
        
        # Overall assessment
        if step1_complete and step1_5_complete and not gaps:
            self.logger.info("✅ Step1 and Step1_5 data appear complete")
        else:
            self.logger.warning("⚠️ Data completeness issues detected - proceeding with warnings")
            
        return validation_result
    
    def _find_step1_files(self, symbol: str, exchange: str) -> Dict[str, str]:
        """Find step1 data collection files."""
        files = {}
        
        # Look for klines data
        klines_patterns = [
            f"klines_{exchange}_{symbol}_1m_consolidated.parquet",
            f"klines_{exchange}_{symbol}_5m_consolidated.parquet", 
            f"klines_{exchange}_{symbol}_15m_consolidated.parquet",
            f"klines_{exchange}_{symbol}_30m_consolidated.parquet",
            f"klines_{exchange}_{symbol}_1h_consolidated.parquet",
            f"klines_{exchange}_{symbol}_4h_consolidated.parquet",
            f"klines_{exchange}_{symbol}_1d_consolidated.parquet"
        ]
        
        for pattern in klines_patterns:
            file_path = self.data_cache_dir / pattern
            if file_path.exists():
                files[pattern] = str(file_path)
        
        # Look for aggtrades data
        aggtrades_pattern = f"aggtrades_{exchange}_{symbol}_consolidated.parquet"
        aggtrades_path = self.data_cache_dir / aggtrades_pattern
        if aggtrades_path.exists():
            files[aggtrades_pattern] = str(aggtrades_path)
        
        return files
    
    def _find_step1_5_files(self, symbol: str, exchange: str) -> Dict[str, str]:
        """Find step1_5 data converter files."""
        files = {}
        
        # Look for processed data files
        processed_patterns = [
            f"processed_{exchange}_{symbol}_train.parquet",
            f"processed_{exchange}_{symbol}_validation.parquet", 
            f"processed_{exchange}_{symbol}_test.parquet"
        ]
        
        for pattern in processed_patterns:
            file_path = self.data_cache_dir / pattern
            if file_path.exists():
                files[pattern] = str(file_path)
        
        return files
    
    def _validate_step1_completeness(
        self, 
        files: Dict[str, str], 
        expected_days: int
    ) -> bool:
        """Validate step1 data completeness."""
        if not files:
            return False
        
        # Check if we have the essential files
        has_aggtrades = any("aggtrades_" in f for f in files.keys())
        has_1m_klines = any("klines_" in f and "1m_consolidated" in f for f in files.keys())
        has_5m_klines = any("klines_" in f and "5m_consolidated" in f for f in files.keys())
        
        # We need at least aggtrades + one klines file
        essential_files = sum([has_aggtrades, has_1m_klines, has_5m_klines])
        
        self.logger.debug(f"Step1 validation: aggtrades={has_aggtrades}, 1m={has_1m_klines}, 5m={has_5m_klines}, total={essential_files}")
        
        # We need at least 2 out of 3 essential files (aggtrades + at least one klines file)
        return essential_files >= 2
    
    def _validate_step1_5_completeness(
        self, 
        files: Dict[str, str], 
        expected_days: int
    ) -> bool:
        """Validate step1_5 data completeness."""
        if not files:
            return False
        
        # Check if we have all three processed datasets
        required_files = ["train", "validation", "test"]
        found_files = 0
        
        for required in required_files:
            for filename in files.keys():
                if f"processed_" in filename and required in filename:
                    found_files += 1
                    break
        
        return found_files >= len(required_files)
    
    def _detect_data_gaps(
        self, 
        step1_files: Dict[str, str], 
        step1_5_files: Dict[str, str],
        expected_days: int
    ) -> List[str]:
        """Detect data gaps in existing files."""
        gaps = []
        
        # Check for missing essential step1 files
        if not any("aggtrades_" in f for f in step1_files.keys()):
            gaps.append("Missing aggtrades data")
        
        if not any("klines_" in f and "1m_consolidated" in f for f in step1_files.keys()):
            gaps.append("Missing 1m klines data")
        
        if not any("klines_" in f and "5m_consolidated" in f for f in step1_files.keys()):
            gaps.append("Missing 5m klines data")
        
        # Check for missing step1_5 files
        if not any("processed_" in f and "train" in f for f in step1_5_files.keys()):
            gaps.append("Missing processed training data")
        
        if not any("processed_" in f and "validation" in f for f in step1_5_files.keys()):
            gaps.append("Missing processed validation data")
        
        if not any("processed_" in f and "test" in f for f in step1_5_files.keys()):
            gaps.append("Missing processed test data")
        
        return gaps
    
    def can_start_from_step2(self, symbol: str, exchange: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if we can start from step2 with existing data.
        
        Returns:
            Tuple of (can_start, validation_result)
        """
        validation_result = self.validate_step1_data_completeness(symbol, exchange)
        
        # Can start if we have both step1 and step1_5 data
        can_start = validation_result["step1_complete"] and validation_result["step1_5_complete"]
        
        return can_start, validation_result
    
    def print_validation_report(self, validation_result: Dict[str, Any], symbol: str, exchange: str):
        """Print a formatted validation report."""
        print("\n" + "="*80)
        print(f"📊 DATA COMPLETENESS VALIDATION REPORT")
        print(f"🎯 Symbol: {symbol}")
        print(f"🏢 Exchange: {exchange}")
        print("="*80)
        
        # Step1 status
        step1_status = "✅ COMPLETE" if validation_result["step1_complete"] else "❌ INCOMPLETE"
        print(f"📁 Step1 Data Collection: {step1_status}")
        
        if validation_result["data_files"].get("step1"):
            print(f"   📄 Found {len(validation_result['data_files']['step1'])} step1 files")
        
        # Step1_5 status
        step1_5_status = "✅ COMPLETE" if validation_result["step1_5_complete"] else "❌ INCOMPLETE"
        print(f"🔄 Step1_5 Data Converter: {step1_5_status}")
        
        if validation_result["data_files"].get("step1_5"):
            print(f"   📄 Found {len(validation_result['data_files']['step1_5'])} step1_5 files")
        
        # Warnings
        if validation_result["warnings"]:
            print(f"\n⚠️  WARNINGS:")
            for warning in validation_result["warnings"]:
                print(f"   • {warning}")
        
        # Gaps
        if validation_result["gaps"]:
            print(f"\n🕳️  DATA GAPS:")
            for gap in validation_result["gaps"]:
                print(f"   • {gap}")
        
        # Recommendations
        if validation_result["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in validation_result["recommendations"]:
                print(f"   • {rec}")
        
        # Overall assessment
        can_start = validation_result["step1_complete"] and validation_result["step1_5_complete"]
        if can_start:
            print(f"\n✅ READY TO START FROM STEP2")
            print(f"   Proceeding with existing data...")
        else:
            print(f"\n❌ NOT READY FOR STEP2")
            print(f"   Missing required data files")
        
        print("="*80 + "\n")


def validate_data_for_step2(symbol: str, exchange: str, data_cache_dir: str = "data_cache") -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to validate data completeness for step2.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_cache_dir: Directory containing data files
        
    Returns:
        Tuple of (can_start_from_step2, validation_result)
    """
    validator = DataCompletenessValidator(data_cache_dir)
    return validator.can_start_from_step2(symbol, exchange)


def print_data_validation_report(symbol: str, exchange: str, data_cache_dir: str = "data_cache"):
    """
    Convenience function to print a data validation report.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_cache_dir: Directory containing data files
    """
    validator = DataCompletenessValidator(data_cache_dir)
    validation_result = validator.validate_step1_data_completeness(symbol, exchange)
    validator.print_validation_report(validation_result, symbol, exchange)