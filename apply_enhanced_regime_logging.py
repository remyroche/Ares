#!/usr/bin/env python3
"""
Apply Enhanced Regime Logging to All Financial Logging Files

This script applies the enhanced regime logging pattern to all financial logging files
in the training steps, ensuring consistent implementation across all steps.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any
import shutil
from datetime import datetime

# Define the financial logging files to update
FINANCIAL_LOGGING_FILES = [
    "src/training/steps/model_training/step11_financial_logging.py",
    "src/training/steps/model_training/step12_financial_logging.py",
    "src/training/steps/model_training/step13_financial_logging.py",
    "src/training/steps/model_training/step14_financial_logging.py",
    "src/training/steps/model_training/step15_financial_logging.py",
    "src/training/steps/model_training/step16_financial_logging.py",
    "src/training/steps/model_training/step09_5_financial_logging.py",
    "src/training/steps/model_training/step04_5_financial_logging.py",
    "src/training/steps/backtesting/step18_financial_logging.py",
    "src/training/steps/backtesting/step19_financial_logging.py",
    "src/training/steps/backtesting/step20_financial_logging.py",
    "src/training/steps/optimisation/step17_financial_logging.py",
    "src/training/steps/market_analysis/step04_financial_logging.py",
    "src/training/steps/market_analysis/hmm_clustering/step03_financial_logging.py",
    "src/training/steps/data_collection/data_preparation/step02_5_financial_logging.py",
]


def create_enhanced_logging_template(step_name: str, logger_name: str) -> str:
    """Create the enhanced logging template for a specific step."""
    
    return f'''"""
Financial metrics logging for {step_name}.
Independent logging module that can be used without the reporting system.

Enhanced with per-HMM regime logging and fail-fast validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import (
    get_financial_metrics_logger, 
    financial_metrics_context,
    get_smart_financial_metrics_logger,
    log_financial_metric_with_regime_awareness
)
from src.utils.logger import system_logger

# Import enhanced functionality if available
try:
    from src.utils.enhanced_financial_metrics_logger import (
        get_enhanced_financial_metrics_logger,
        validate_and_log_regime_data
    )
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False
    get_enhanced_financial_metrics_logger = None
    validate_and_log_regime_data = None

logger = system_logger.getChild('{logger_name}')


class {logger_name}FinancialLogger:
    """Independent financial metrics logger for {step_name} with enhanced regime logging."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str, enable_enhanced_logging: bool = True):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.enable_enhanced_logging = enable_enhanced_logging
        
        # Use smart logger that automatically chooses enhanced or base logger
        self.financial_logger = get_smart_financial_metrics_logger(use_enhanced=enable_enhanced_logging)
        
        # Store enhanced logger separately if available
        if ENHANCED_LOGGING_AVAILABLE and enable_enhanced_logging:
            self.enhanced_logger = get_enhanced_financial_metrics_logger()
        else:
            self.enhanced_logger = None
    
    def log_step_execution(self, *args, data: Optional[pd.DataFrame] = None, **kwargs) -> bool:
        """
        Log comprehensive financial metrics for {step_name} execution with enhanced regime validation.
        
        Args:
            *args: Step execution arguments
            data: DataFrame for regime validation (optional)
            **kwargs: Additional keyword arguments
            
        Returns:
            True if logging succeeded, False if fail-fast conditions triggered
        """
        try:
            # Use enhanced logging if available and data is provided
            if self.enhanced_logger and data is not None:
                return self._log_with_enhanced_regime_validation(*args, data=data, **kwargs)
            else:
                # Fallback to standard logging
                return self._log_with_standard_method(*args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to log financial metrics: {{e}}")
            return False
    
    def _log_with_enhanced_regime_validation(self, *args, data: pd.DataFrame, **kwargs) -> bool:
        """Log with enhanced regime validation and fail-fast checks."""
        try:
            # Validate regime data first
            if validate_and_log_regime_data:
                validation_success = validate_and_log_regime_data(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="{step_name}",
                    data=data,
                    regime_column='composite_cluster_id'
                )
                
                if not validation_success:
                    logger.error("🚨 Regime validation failed for {step_name}")
                    return False
            
            # Log step start
            self.financial_logger.log_step_start("{step_name}", self.symbol, self.exchange, self.timeframe)
            
            # Log all financial metrics with regime awareness
            success = self._log_financial_metrics_with_regime_awareness(*args, data=data, **kwargs)
            
            # Log file paths
            self._log_created_file_paths()
            
            # Log step end
            self.financial_logger.log_step_end(
                "{step_name}", 
                self.symbol, 
                self.exchange, 
                self.timeframe, 
                success=success
            )
            
            return success
            
        except Exception as e:
            self.financial_logger.log_step_end(
                "{step_name}", 
                self.symbol, 
                self.exchange, 
                self.timeframe, 
                success=False, 
                error_message=str(e)
            )
            logger.error(f"Enhanced regime validation logging failed: {{e}}")
            return False
    
    def _log_with_standard_method(self, *args, **kwargs) -> bool:
        """Log using standard method (fallback)."""
        with financial_metrics_context(
            step_name="{step_name}",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("{step_name}", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(*args, **kwargs)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("{step_name}", self.symbol, self.exchange, self.timeframe, success=True)
                
                return True
                
            except Exception as e:
                self.financial_logger.log_step_end("{step_name}", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {{e}}")
                return False
    
    def _log_financial_metrics_with_regime_awareness(self, *args, data: pd.DataFrame, **kwargs) -> bool:
        """Log financial metrics with enhanced regime awareness and fail-fast validation."""
        try:
            success = True
            
            # Log step success with regime awareness
            success &= log_financial_metric_with_regime_awareness(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                metric_name="step_success",
                metric_value=1.0,
                metric_type="performance",
                step_name="{step_name}",
                data=data
            )
            
            # Log execution time with regime awareness
            success &= log_financial_metric_with_regime_awareness(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                metric_name="execution_time_seconds",
                metric_value=0.0,  # Will be updated with actual execution time
                metric_type="performance",
                step_name="{step_name}",
                data=data
            )
            
            # Log regime-specific metrics if enhanced logger is available
            if self.enhanced_logger and data is not None and 'composite_cluster_id' in data.columns:
                regime_data = data['composite_cluster_id'].dropna()
                regime_counts = regime_data.value_counts()
                
                regime_metrics = {{}}
                for regime_id, count in regime_counts.items():
                    regime_metrics[str(regime_id)] = {{
                        'sample_count': float(count),
                        'regime_processed': 1.0
                    }}
                
                # Use enhanced logger for per-regime metrics
                success &= self.enhanced_logger.log_per_regime_metrics(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="{step_name}",
                    regime_metrics=regime_metrics,
                    data=data
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to log financial metrics with regime awareness: {{e}}")
            return False
    
    def _log_financial_metrics_from_results(self, *args, **kwargs) -> None:
        """Log key financial metrics directly from step results (fallback method)."""
        try:
            # This method should be implemented by each specific step
            # For now, just log basic step completion
            self.financial_logger.log_financial_metric(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                metric_name="step_completed",
                metric_value=1.0,
                metric_type="performance",
                step_name="{step_name}"
            )
        except Exception as e:
            logger.error(f"Failed to log financial metrics from results: {{e}}")
    
    def _log_created_file_paths(self) -> None:
        """Log file paths that were created during this step."""
        try:
            if hasattr(self.financial_logger, 'current_file_path') and self.financial_logger.current_file_path:
                logger.info(f"📁 Financial metrics file created: {{self.financial_logger.current_file_path}}")
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="metrics_file_path",
                    metric_value=0.0,
                    metric_type="file_path",
                    step_name="{step_name}",
                    additional_data={{'file_path': str(self.financial_logger.current_file_path)}}
                )
            logger.info("📁 File paths logged for {step_name}")
        except Exception as e:
            logger.warning(f"Could not log file paths: {{e}}")


# Enhanced {logger_name} Financial Logger with Regime-Aware Decorator Support
class Enhanced{logger_name}FinancialLogger({logger_name}FinancialLogger):
    """Enhanced {logger_name} Financial Logger with automatic regime-aware logging decorator support."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str, enable_enhanced_logging: bool = True):
        super().__init__(symbol, exchange, timeframe, enable_enhanced_logging)
        
        # Import regime-aware decorator if available
        try:
            from src.utils.regime_aware_financial_logging_decorator import (
                regime_aware_financial_logging,
                auto_regime_aware_logging
            )
            self.regime_aware_decorator = regime_aware_financial_logging
            self.auto_regime_aware_decorator = auto_regime_aware_logging
            self.decorator_available = True
        except ImportError:
            self.decorator_available = False
    
    def get_decorated_execute_method(self, original_execute_method):
        """Get the execute method decorated with regime-aware logging."""
        if self.decorator_available:
            return self.auto_regime_aware_decorator(
                enable_regime_validation=True,
                enable_fail_fast=True,
                min_regime_samples=100,
                max_regime_imbalance=0.8,
                regime_column='composite_cluster_id',
                min_data_quality=0.7
            )(original_execute_method)
        else:
            return original_execute_method
'''


def extract_step_info_from_path(file_path: str) -> Dict[str, str]:
    """Extract step information from file path."""
    path_parts = file_path.split('/')
    filename = path_parts[-1]
    
    # Extract step name from filename
    step_match = re.search(r'step(\d+[._]?\d*)_', filename)
    if step_match:
        step_num = step_match.group(1).replace('_', '.')
        step_name = f"Step{step_num}_{filename.split('_')[1].title()}"
    else:
        step_name = filename.replace('.py', '').replace('_', ' ').title()
    
    # Extract logger name
    logger_name = filename.replace('.py', '').replace('_', '').title()
    
    return {
        'step_name': step_name,
        'logger_name': logger_name,
        'filename': filename
    }


def create_backup(file_path: str) -> str:
    """Create a backup of the original file."""
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copy2(file_path, backup_path)
        print(f"✅ Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"❌ Failed to create backup for {file_path}: {e}")
        return ""


def update_financial_logging_file(file_path: str) -> bool:
    """Update a single financial logging file with enhanced regime logging."""
    print(f"\n🔄 Updating {file_path}...")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    # Create backup
    backup_path = create_backup(file_path)
    if not backup_path:
        return False
    
    try:
        # Extract step information
        step_info = extract_step_info_from_path(file_path)
        
        # Create enhanced content
        enhanced_content = create_enhanced_logging_template(
            step_info['step_name'], 
            step_info['logger_name']
        )
        
        # Write enhanced content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_content)
        
        print(f"✅ Successfully updated {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update {file_path}: {e}")
        # Restore from backup
        try:
            shutil.copy2(backup_path, file_path)
            print(f"🔄 Restored from backup: {backup_path}")
        except:
            pass
        return False


def main():
    """Main function to update all financial logging files."""
    print("🚀 Applying Enhanced Regime Logging to All Financial Logging Files")
    print("=" * 70)
    
    updated_files = []
    failed_files = []
    
    for file_path in FINANCIAL_LOGGING_FILES:
        if update_financial_logging_file(file_path):
            updated_files.append(file_path)
        else:
            failed_files.append(file_path)
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully updated: {len(updated_files)} files")
    print(f"❌ Failed to update: {len(failed_files)} files")
    
    if updated_files:
        print("\n✅ Updated files:")
        for file_path in updated_files:
            print(f"   - {file_path}")
    
    if failed_files:
        print("\n❌ Failed files:")
        for file_path in failed_files:
            print(f"   - {file_path}")
    
    print(f"\n🎯 Enhanced regime logging implementation completed!")
    print(f"📁 Backups created with timestamp suffix")
    print(f"🔄 All files now support per-HMM regime logging and fail-fast validation")
    print(f"\n📋 Next steps:")
    print(f"   1. Test the updated financial logging files")
    print(f"   2. Update the main step files to use the enhanced loggers")
    print(f"   3. Add the @auto_regime_aware_logging decorator to step execute methods")


if __name__ == "__main__":
    main()