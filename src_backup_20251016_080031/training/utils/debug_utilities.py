"""
Enhanced Debugging and Validation Utilities for Training Pipeline

This module provides comprehensive debugging, logging, and validation utilities
to identify and resolve silent failures in the training pipeline.
"""

import os
import sys
import json
import time
import traceback
import logging
import psutil
import importlib
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager
import pandas as pd
import numpy as np

# Import logging utilities
try:
    from src.utils.logger import system_logger
    from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_info, tprint_success, tprint_debug
    LOGGING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced logging not available: {e}")
    LOGGING_AVAILABLE = False
    
    # Fallback logging
    def tprint(msg, **kwargs):
        print(f"[{datetime.now()}] {msg}")
    
    tprint_error = tprint_warning = tprint_info = tprint_success = tprint_debug = tprint

logger = system_logger.getChild('DebugUtilities') if LOGGING_AVAILABLE else logging.getLogger('DebugUtilities')

@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)

@dataclass
class DependencyCheck:
    """Result of dependency validation."""
    package_name: str
    is_available: bool
    version: Optional[str] = None
    error_message: Optional[str] = None
    import_path: Optional[str] = None

class TrainingDebugger:
    """Comprehensive debugging and validation utility for training pipeline."""
    
    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the debugger for a specific training step."""
        self.step_name = step_name
        self.config = config or {}
        self.start_time = datetime.now()
        self.validation_results: List[ValidationResult] = []
        self.dependency_checks: List[DependencyCheck] = []
        
        # Setup enhanced logging
        self.logger = logger.getChild(f'Debug_{step_name}')
        
        tprint_info(f"🔍 Initializing TrainingDebugger for step: {step_name}")
        
    def validate_dependencies(self) -> bool:
        """Validate all required dependencies for training."""
        tprint_info("🔧 Validating dependencies...")
        
        # Core dependencies
        core_deps = [
            ('numpy', 'numpy'),
            ('pandas', 'pandas'),
            ('sklearn', 'sklearn'),
            ('xgboost', 'xgboost'),
            ('lightgbm', 'lightgbm'),
        ]
        
        # ML dependencies
        ml_deps = [
            ('hmmlearn', 'hmmlearn'),
            ('optuna', 'optuna'),
            ('joblib', 'joblib'),
        ]
        
        # System dependencies
        system_deps = [
            ('psutil', 'psutil'),
            ('pathlib', 'pathlib'),
        ]
        
        all_deps = core_deps + ml_deps + system_deps
        all_available = True
        
        for pkg_name, import_name in all_deps:
            check = self._check_dependency(pkg_name, import_name)
            self.dependency_checks.append(check)
            
            if check.is_available:
                tprint_success(f"  ✅ {pkg_name}: {check.version or 'Available'}")
            else:
                tprint_error(f"  ❌ {pkg_name}: {check.error_message}")
                all_available = False
        
        if all_available:
            tprint_success("🎉 All dependencies validated successfully!")
        else:
            tprint_error("❌ Dependency validation failed!")
            
        return all_available
    
    def _check_dependency(self, package_name: str, import_name: str) -> DependencyCheck:
        """Check if a specific dependency is available."""
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', None)
            return DependencyCheck(
                package_name=package_name,
                is_available=True,
                version=version,
                import_path=import_name
            )
        except ImportError as e:
            return DependencyCheck(
                package_name=package_name,
                is_available=False,
                error_message=str(e),
                import_path=import_name
            )
    
    def validate_data_files(self, data_dir: str, symbol: str, timeframe: str, exchange: str = "binance") -> ValidationResult:
        """Validate that required data files are accessible."""
        tprint_info(f"📂 Validating data files for {symbol} {timeframe}...")
        
        try:
            data_path = Path(data_dir)
            if not data_path.exists():
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Data directory does not exist: {data_path}",
                    suggestions=["Create the data directory", "Check data_dir configuration"]
                )
            
            # Check for various data file patterns
            patterns_to_check = [
                f"{exchange}/{symbol.lower()}/{timeframe}/*.parquet",
                f"{exchange}/{symbol}/{timeframe}/*.parquet",
                f"{symbol}_{timeframe}_*.parquet",
                f"{symbol.lower()}_{timeframe}_*.parquet",
            ]
            
            found_files = []
            for pattern in patterns_to_check:
                files = list(data_path.glob(pattern))
                found_files.extend(files)
            
            if not found_files:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"No data files found for {symbol} {timeframe} in {data_path}",
                    details={
                        "data_dir": str(data_path),
                        "patterns_checked": patterns_to_check,
                        "directory_contents": list(data_path.rglob("*"))[:20]  # First 20 files
                    },
                    suggestions=[
                        "Run data collection pipeline first",
                        "Check symbol and timeframe spelling",
                        "Verify data directory structure"
                    ]
                )
            
            # Validate file accessibility and basic structure
            accessible_files = []
            for file_path in found_files[:5]:  # Check first 5 files
                try:
                    if file_path.suffix == '.parquet':
                        # Read parquet file without nrows parameter (compatibility issue)
                        df = pd.read_parquet(file_path)
                        df_sample = df.head(10)  # Get first 10 rows after reading
                        accessible_files.append({
                            "file": str(file_path),
                            "size": file_path.stat().st_size,
                            "rows_sample": len(df_sample),
                            "total_rows": len(df),
                            "columns": list(df.columns)
                        })
                    else:
                        accessible_files.append({
                            "file": str(file_path),
                            "size": file_path.stat().st_size,
                            "type": "non-parquet"
                        })
                except Exception as e:
                    tprint_warning(f"  ⚠️ Cannot read file {file_path}: {e}")
            
            tprint_success(f"  ✅ Found {len(found_files)} data files, {len(accessible_files)} accessible")
            
            return ValidationResult(
                is_valid=True,
                details={
                    "total_files": len(found_files),
                    "accessible_files": len(accessible_files),
                    "sample_files": accessible_files
                }
            )
            
        except Exception as e:
            tprint_error(f"  ❌ Error validating data files: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Exception during data file validation: {str(e)}",
                details={"traceback": traceback.format_exc()}
            )
    
    def validate_training_data(self, data: Optional[pd.DataFrame], min_rows: int = 1000) -> ValidationResult:
        """Validate training data quality and structure."""
        tprint_info("📊 Validating training data...")
        
        if data is None:
            return ValidationResult(
                is_valid=False,
                error_message="Training data is None",
                suggestions=["Check data loading process", "Verify data file accessibility"]
            )
        
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                error_message=f"Training data is not a DataFrame, got {type(data)}",
                suggestions=["Check data loading and conversion process"]
            )
        
        if data.empty:
            return ValidationResult(
                is_valid=False,
                error_message="Training data is empty",
                suggestions=["Check data filtering logic", "Verify data file contents"]
            )
        
        if len(data) < min_rows:
            return ValidationResult(
                is_valid=False,
                error_message=f"Insufficient training data: {len(data)} rows < {min_rows} required",
                details={"actual_rows": len(data), "required_rows": min_rows},
                suggestions=["Increase data collection period", "Reduce minimum data requirements"]
            )
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            return ValidationResult(
                is_valid=False,
                error_message=f"Missing required columns: {missing_columns}",
                details={
                    "available_columns": list(data.columns),
                    "missing_columns": missing_columns
                },
                suggestions=["Check data preprocessing pipeline", "Verify column naming conventions"]
            )
        
        # Check for data quality issues
        quality_issues = []
        
        # Check for NaN values
        nan_columns = data.columns[data.isnull().any()].tolist()
        if nan_columns:
            quality_issues.append(f"NaN values in columns: {nan_columns}")
        
        # Check for infinite values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        inf_columns = []
        for col in numeric_columns:
            if np.isinf(data[col]).any():
                inf_columns.append(col)
        if inf_columns:
            quality_issues.append(f"Infinite values in columns: {inf_columns}")
        
        # Check for constant columns
        constant_columns = []
        for col in numeric_columns:
            if data[col].nunique() <= 1:
                constant_columns.append(col)
        if constant_columns:
            quality_issues.append(f"Constant columns: {constant_columns}")
        
        tprint_success(f"  ✅ Training data validation: {len(data)} rows, {len(data.columns)} columns")
        if quality_issues:
            tprint_warning(f"  ⚠️ Data quality issues: {'; '.join(quality_issues)}")
        
        return ValidationResult(
            is_valid=len(quality_issues) == 0,
            error_message="; ".join(quality_issues) if quality_issues else None,
            details={
                "rows": len(data),
                "columns": len(data.columns),
                "column_names": list(data.columns),
                "data_types": data.dtypes.to_dict(),
                "quality_issues": quality_issues
            }
        )
    
    def validate_system_resources(self, min_memory_gb: float = 1.0, min_disk_gb: float = 1.0) -> ValidationResult:
        """Validate system resources are sufficient for training."""
        tprint_info("💻 Validating system resources...")
        
        try:
            # Memory check
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
            
            # Disk check
            disk = psutil.disk_usage('/')
            available_disk_gb = disk.free / (1024**3)
            
            # CPU check
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            issues = []
            if available_memory_gb < min_memory_gb:
                issues.append(f"Low memory: {available_memory_gb:.1f}GB < {min_memory_gb}GB required")
            
            if available_disk_gb < min_disk_gb:
                issues.append(f"Low disk space: {available_disk_gb:.1f}GB < {min_disk_gb}GB required")
            
            tprint_success(f"  ✅ Memory: {available_memory_gb:.1f}GB, Disk: {available_disk_gb:.1f}GB, CPU: {cpu_count} cores")
            
            return ValidationResult(
                is_valid=len(issues) == 0,
                error_message="; ".join(issues) if issues else None,
                details={
                    "memory_gb": available_memory_gb,
                    "disk_gb": available_disk_gb,
                    "cpu_cores": cpu_count,
                    "cpu_usage_percent": cpu_percent
                }
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Error checking system resources: {str(e)}",
                details={"traceback": traceback.format_exc()}
            )
    
    @contextmanager
    def debug_context(self, operation_name: str):
        """Context manager for debugging operations with comprehensive error capture."""
        operation_start = time.time()
        tprint_info(f"🚀 Starting operation: {operation_name}")
        
        try:
            yield
            duration = time.time() - operation_start
            tprint_success(f"✅ Completed operation: {operation_name} in {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - operation_start
            error_details = {
                "operation": operation_name,
                "step": self.step_name,
                "duration_seconds": duration,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "system_info": self._get_system_info(),
                "config": self.config
            }
            
            tprint_error(f"❌ FAILED operation: {operation_name} after {duration:.2f}s")
            tprint_error(f"   Error: {type(e).__name__}: {str(e)}")
            
            # Save detailed error report
            self._save_error_report(error_details)
            
            # Re-raise the exception with additional context
            raise RuntimeError(f"Operation '{operation_name}' failed in step '{self.step_name}': {str(e)}") from e
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information for debugging."""
        try:
            return {
                "python_version": sys.version,
                "platform": sys.platform,
                "memory_usage": psutil.virtual_memory()._asdict(),
                "cpu_info": {
                    "count": psutil.cpu_count(),
                    "usage": psutil.cpu_percent(interval=0.1)
                },
                "disk_usage": psutil.disk_usage('/')._asdict(),
                "current_directory": os.getcwd(),
                "environment_variables": dict(os.environ)
            }
        except Exception as e:
            return {"error": f"Failed to get system info: {str(e)}"}
    
    def _save_error_report(self, error_details: Dict[str, Any]):
        """Save detailed error report for debugging."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = Path(f"debug_reports/error_report_{self.step_name}_{timestamp}.json")
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(error_details, f, indent=2, default=str)
            
            tprint_info(f"📝 Error report saved to: {report_path}")
            
        except Exception as e:
            tprint_error(f"Failed to save error report: {e}")
    
    def comprehensive_validation(self, 
                                data: Optional[pd.DataFrame] = None,
                                data_dir: Optional[str] = None,
                                symbol: Optional[str] = None,
                                timeframe: Optional[str] = None,
                                exchange: str = "binance") -> bool:
        """Run comprehensive validation of all components."""
        tprint_info("🔍 Running comprehensive validation...")
        
        all_valid = True
        
        # 1. Validate dependencies
        if not self.validate_dependencies():
            all_valid = False
        
        # 2. Validate system resources
        resource_result = self.validate_system_resources()
        if not resource_result.is_valid:
            tprint_error(f"❌ System resources: {resource_result.error_message}")
            all_valid = False
        
        # 3. Validate data files if parameters provided
        if data_dir and symbol and timeframe:
            file_result = self.validate_data_files(data_dir, symbol, timeframe, exchange)
            if not file_result.is_valid:
                tprint_error(f"❌ Data files: {file_result.error_message}")
                all_valid = False
        
        # 4. Validate training data if provided
        if data is not None:
            data_result = self.validate_training_data(data)
            if not data_result.is_valid:
                tprint_error(f"❌ Training data: {data_result.error_message}")
                all_valid = False
        
        if all_valid:
            tprint_success("🎉 All validations passed!")
        else:
            tprint_error("❌ Validation failures detected!")
        
        return all_valid
    
    def create_debug_report(self) -> Dict[str, Any]:
        """Create comprehensive debug report."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "step_name": self.step_name,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": duration,
            "validation_results": [
                {
                    "is_valid": r.is_valid,
                    "error_message": r.error_message,
                    "details": r.details,
                    "suggestions": r.suggestions
                }
                for r in self.validation_results
            ],
            "dependency_checks": [
                {
                    "package_name": d.package_name,
                    "is_available": d.is_available,
                    "version": d.version,
                    "error_message": d.error_message
                }
                for d in self.dependency_checks
            ],
            "system_info": self._get_system_info(),
            "config": self.config
        }


def create_enhanced_error_handler(step_name: str) -> Callable:
    """Create an enhanced error handler decorator for training steps."""
    
    def error_handler_decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            debugger = TrainingDebugger(step_name)
            
            try:
                with debugger.debug_context(f"execute_{step_name}"):
                    # Pre-execution validation
                    tprint_info(f"🔍 Pre-execution validation for {step_name}")
                    
                    # Extract common parameters for validation
                    config = kwargs.get('config', {})
                    data_dir = config.get('data_dir', 'historical_data')
                    symbol = config.get('symbol', 'ETHUSDT')
                    timeframe = config.get('timeframe', '1m')
                    
                    # Run comprehensive validation
                    if not debugger.comprehensive_validation(
                        data_dir=data_dir,
                        symbol=symbol,
                        timeframe=timeframe
                    ):
                        raise RuntimeError(f"Pre-execution validation failed for {step_name}")
                    
                    # Execute the actual function
                    result = func(*args, **kwargs)
                    
                    # Post-execution validation
                    tprint_info(f"✅ {step_name} completed successfully")
                    return result
                    
            except Exception as e:
                tprint_error(f"❌ {step_name} failed with error: {str(e)}")
                
                # Create and save debug report
                debug_report = debugger.create_debug_report()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = Path(f"debug_reports/{step_name}_failure_{timestamp}.json")
                report_path.parent.mkdir(exist_ok=True)
                
                with open(report_path, 'w') as f:
                    json.dump(debug_report, f, indent=2, default=str)
                
                tprint_error(f"📝 Debug report saved to: {report_path}")
                
                # Re-raise with enhanced context
                raise RuntimeError(f"{step_name} failed: {str(e)}") from e
        
        return wrapper
    return error_handler_decorator


# Utility functions for quick validation
def quick_dependency_check() -> bool:
    """Quick check of all critical dependencies."""
    debugger = TrainingDebugger("quick_check")
    return debugger.validate_dependencies()

def quick_data_check(data_dir: str, symbol: str, timeframe: str, exchange: str = "binance") -> bool:
    """Quick check of data file accessibility."""
    debugger = TrainingDebugger("quick_data_check")
    result = debugger.validate_data_files(data_dir, symbol, timeframe, exchange)
    return result.is_valid

def quick_system_check() -> bool:
    """Quick check of system resources."""
    debugger = TrainingDebugger("quick_system_check")
    result = debugger.validate_system_resources()
    return result.is_valid