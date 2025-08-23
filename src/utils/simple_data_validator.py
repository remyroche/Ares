"""
Simple Data Validator

A lightweight data validator that checks file existence for step1 and step1_5
without requiring pandas/numpy dependencies.
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class SimpleDataValidator:
    """Simple validator that checks file existence for step readiness."""
    
    def __init__(self, data_cache_dir: str = "data_cache"):
        self.data_cache_dir = Path(data_cache_dir)
        self.logger = logging.getLogger(f"{__name__}.SimpleDataValidator")
    
    def validate_step1_files(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Validate Step1 file existence.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'BINANCE')
            
        Returns:
            Dict with validation results
        """
        self.logger.info(f"🔍 Checking Step1 files for {symbol} on {exchange}")
        
        results = {
            "step": "step1_data_collection",
            "symbol": symbol,
            "exchange": exchange,
            "validation_passed": False,
            "issues": [],
            "file_checks": {}
        }
        
        # Required step1 files
        required_files = [
            f"klines_{exchange}_{symbol}_1m_consolidated.parquet",
            f"aggtrades_{exchange}_{symbol}_consolidated.parquet"
        ]
        
        # Optional but helpful files
        optional_files = [
            f"klines_{exchange}_{symbol}_5m_consolidated.parquet",
            f"klines_{exchange}_{symbol}_15m_consolidated.parquet"
        ]
        
        # Check required files
        required_found = 0
        for filename in required_files:
            file_path = self.data_cache_dir / filename
            exists = file_path.exists()
            
            results["file_checks"][filename] = {
                "exists": exists,
                "path": str(file_path),
                "required": True
            }
            
            if exists:
                required_found += 1
                self.logger.info(f"   ✅ Found: {filename}")
            else:
                results["issues"].append(f"Required file missing: {filename}")
                self.logger.warning(f"   ❌ Missing: {filename}")
        
        # Check optional files
        optional_found = 0
        for filename in optional_files:
            file_path = self.data_cache_dir / filename
            exists = file_path.exists()
            
            results["file_checks"][filename] = {
                "exists": exists,
                "path": str(file_path),
                "required": False
            }
            
            if exists:
                optional_found += 1
                self.logger.info(f"   ✅ Found: {filename}")
        
        # Validation passes if we have all required files
        results["validation_passed"] = required_found == len(required_files)
        
        if results["validation_passed"]:
            self.logger.info(f"✅ Step1 file validation passed ({required_found}/{len(required_files)} required files)")
        else:
            self.logger.warning(f"⚠️ Step1 file validation failed ({required_found}/{len(required_files)} required files)")
        
        return results
    
    def validate_step1_5_files(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Validate Step1_5 file existence.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'BINANCE')
            
        Returns:
            Dict with validation results
        """
        self.logger.info(f"🔍 Checking Step1_5 files for {symbol} on {exchange}")
        
        results = {
            "step": "step1_5_data_converter",
            "symbol": symbol,
            "exchange": exchange,
            "validation_passed": False,
            "issues": [],
            "file_checks": {}
        }
        
        # Required step1_5 files
        required_files = [
            f"processed_{exchange}_{symbol}_train.parquet",
            f"processed_{exchange}_{symbol}_validation.parquet",
            f"processed_{exchange}_{symbol}_test.parquet"
        ]
        
        # Check required files
        required_found = 0
        for filename in required_files:
            file_path = self.data_cache_dir / filename
            exists = file_path.exists()
            
            results["file_checks"][filename] = {
                "exists": exists,
                "path": str(file_path),
                "required": True
            }
            
            if exists:
                required_found += 1
                self.logger.info(f"   ✅ Found: {filename}")
            else:
                results["issues"].append(f"Required file missing: {filename}")
                self.logger.warning(f"   ❌ Missing: {filename}")
        
        # Validation passes if we have all required files
        results["validation_passed"] = required_found == len(required_files)
        
        if results["validation_passed"]:
            self.logger.info(f"✅ Step1_5 file validation passed ({required_found}/{len(required_files)} required files)")
        else:
            self.logger.warning(f"⚠️ Step1_5 file validation failed ({required_found}/{len(required_files)} required files)")
        
        return results
    
    def can_start_step2(self, symbol: str, exchange: str) -> tuple[bool, dict]:
        """
        Check if we can start step2 with existing data.
        
        Returns:
            Tuple of (can_start, validation_results)
        """
        step1_result = self.validate_step1_files(symbol, exchange)
        step1_5_result = self.validate_step1_5_files(symbol, exchange)
        
        can_start = (
            step1_result.get("validation_passed", False) and 
            step1_5_result.get("validation_passed", False)
        )
        
        validation_results = {
            "can_start_step2": can_start,
            "step1": step1_result,
            "step1_5": step1_5_result
        }
        
        return can_start, validation_results


def validate_step1_files(symbol: str, exchange: str, data_cache_dir: str = "data_cache") -> Dict[str, Any]:
    """Convenience function to validate Step1 files."""
    validator = SimpleDataValidator(data_cache_dir)
    return validator.validate_step1_files(symbol, exchange)


def validate_step1_5_files(symbol: str, exchange: str, data_cache_dir: str = "data_cache") -> Dict[str, Any]:
    """Convenience function to validate Step1_5 files."""
    validator = SimpleDataValidator(data_cache_dir)
    return validator.validate_step1_5_files(symbol, exchange)


def can_start_step2(symbol: str, exchange: str, data_cache_dir: str = "data_cache") -> tuple[bool, dict]:
    """Convenience function to check if step2 can start."""
    validator = SimpleDataValidator(data_cache_dir)
    return validator.can_start_step2(symbol, exchange)