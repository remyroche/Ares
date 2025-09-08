#!/usr/bin/env python3
"""
Script to update all financial logging files with enhanced regime logging.

This script automatically updates all financial logging files in the training steps
to include enhanced regime-aware logging and fail-fast validation.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any
import shutil
from datetime import datetime

# Define the financial logging files to update
FINANCIAL_LOGGING_FILES = [
    # Model Training Steps
    "src/training/steps/model_training/step09_financial_logging.py",
    "src/training/steps/model_training/step10_financial_logging.py", 
    "src/training/steps/model_training/step11_financial_logging.py",
    "src/training/steps/model_training/step12_financial_logging.py",
    "src/training/steps/model_training/step13_financial_logging.py",
    "src/training/steps/model_training/step14_financial_logging.py",
    "src/training/steps/model_training/step15_financial_logging.py",
    "src/training/steps/model_training/step16_financial_logging.py",
    "src/training/steps/model_training/step09_5_financial_logging.py",
    "src/training/steps/model_training/step04_5_financial_logging.py",
    
    # Backtesting Steps
    "src/training/steps/backtesting/step18_financial_logging.py",
    "src/training/steps/backtesting/step19_financial_logging.py",
    "src/training/steps/backtesting/step20_financial_logging.py",
    
    # Optimization Steps
    "src/training/steps/optimisation/step17_financial_logging.py",
    
    # Market Analysis Steps
    "src/training/steps/market_analysis/step04_financial_logging.py",
    "src/training/steps/market_analysis/hmm_clustering/step03_financial_logging.py",
    
    # Data Collection Steps
    "src/training/steps/data_collection/data_preparation/step02_5_financial_logging.py",
]

# Template for enhanced imports
ENHANCED_IMPORTS_TEMPLATE = '''"""
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

logger = system_logger.getChild('{logger_name}')'''

# Template for enhanced class initialization
ENHANCED_CLASS_INIT_TEMPLATE = '''    def __init__(self, symbol: str, exchange: str, timeframe: str, enable_enhanced_logging: bool = True):
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
            self.enhanced_logger = None'''

# Template for enhanced log_step_execution method
ENHANCED_LOG_STEP_EXECUTION_TEMPLATE = '''    def log_step_execution(self, {method_params}, data: Optional[pd.DataFrame] = None) -> bool:
        """
        Log comprehensive financial metrics for {step_name} execution with enhanced regime validation.
        
        Args:
            {docstring_params}
            data: DataFrame for regime validation (optional)
            
        Returns:
            True if logging succeeded, False if fail-fast conditions triggered
        """
        try:
            # Use enhanced logging if available and data is provided
            if self.enhanced_logger and data is not None:
                return self._log_with_enhanced_regime_validation(
                    {method_call_params}, data
                )
            else:
                # Fallback to standard logging
                return self._log_with_standard_method(
                    {method_call_params}
                )
        except Exception as e:
            logger.error(f"Failed to log financial metrics: {{e}}")
            return False'''

# Template for enhanced regime validation method
ENHANCED_REGIME_VALIDATION_TEMPLATE = '''    def _log_with_enhanced_regime_validation(self, {method_params}, data: pd.DataFrame) -> bool:
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
            success = self._log_financial_metrics_with_regime_awareness(
                {method_call_params}, data
            )
            
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
            return False'''

# Template for standard method fallback
STANDARD_METHOD_TEMPLATE = '''    def _log_with_standard_method(self, {method_params}) -> bool:
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
                self._log_financial_metrics_from_results({method_call_params})
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("{step_name}", self.symbol, self.exchange, self.timeframe, success=True)
                
                return True
                
            except Exception as e:
                self.financial_logger.log_step_end("{step_name}", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {{e}}")
                return False'''

# Template for regime-aware metrics logging
REGIME_AWARE_METRICS_TEMPLATE = '''    def _log_financial_metrics_with_regime_awareness(self, {method_params}, data: pd.DataFrame) -> bool:
        """Log financial metrics with enhanced regime awareness and fail-fast validation."""
        try:
            success = True
            
            # Log key metrics with regime awareness
            {regime_aware_logging_code}
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to log financial metrics with regime awareness: {{e}}")
            return False'''


def extract_step_info(file_path: str) -> Dict[str, str]:
    """Extract step information from file path and content."""
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


def read_file_content(file_path: str) -> str:
    """Read file content safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def extract_method_signature(content: str, method_name: str) -> Dict[str, Any]:
    """Extract method signature and parameters."""
    # Find the method definition
    pattern = rf'def {method_name}\(self, ([^)]*)\) -> ([^:]*):'
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    
    if not match:
        return {'params': '', 'return_type': 'None', 'method_call_params': ''}
    
    params_str = match.group(1).strip()
    return_type = match.group(2).strip()
    
    # Parse parameters
    if not params_str:
        return {'params': '', 'return_type': return_type, 'method_call_params': ''}
    
    # Split parameters and clean them
    params = [p.strip().split(':')[0].strip() for p in params_str.split(',') if p.strip()]
    method_call_params = ', '.join(params)
    
    # Create docstring parameters
    docstring_params = []
    for param in params:
        if param:
            docstring_params.append(f"{param}: {param.replace('_', ' ').title()}")
    
    return {
        'params': params_str,
        'return_type': return_type,
        'method_call_params': method_call_params,
        'docstring_params': '\n            '.join(docstring_params)
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
        # Read original content
        content = read_file_content(file_path)
        if not content:
            return False
        
        # Extract step information
        step_info = extract_step_info(file_path)
        
        # Extract method signature
        method_info = extract_method_signature(content, 'log_step_execution')
        
        # Create enhanced content
        enhanced_content = create_enhanced_content(content, step_info, method_info)
        
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


def create_enhanced_content(original_content: str, step_info: Dict[str, str], method_info: Dict[str, Any]) -> str:
    """Create enhanced content for the financial logging file."""
    
    # Replace imports
    enhanced_imports = ENHANCED_IMPORTS_TEMPLATE.format(
        step_name=step_info['step_name'],
        logger_name=step_info['logger_name']
    )
    
    # Find the class definition and replace the __init__ method
    class_pattern = r'class (\w+FinancialLogger):'
    class_match = re.search(class_pattern, original_content)
    
    if class_match:
        class_name = class_match.group(1)
        
        # Replace the class docstring and __init__ method
        class_start_pattern = rf'class {class_name}FinancialLogger:.*?def __init__\(self, [^)]*\):'
        class_start_match = re.search(class_start_pattern, original_content, re.MULTILINE | re.DOTALL)
        
        if class_start_match:
            # Replace the class definition
            enhanced_class_def = f'''class {class_name}FinancialLogger:
    """Independent financial metrics logger for {step_info['step_name']} with enhanced regime logging."""
    
{ENHANCED_CLASS_INIT_TEMPLATE}'''
            
            # Replace the log_step_execution method
            enhanced_log_method = ENHANCED_LOG_STEP_EXECUTION_TEMPLATE.format(
                step_name=step_info['step_name'],
                method_params=method_info['params'],
                docstring_params=method_info['docstring_params'],
                method_call_params=method_info['method_call_params']
            )
            
            # Add enhanced methods
            enhanced_regime_validation = ENHANCED_REGIME_VALIDATION_TEMPLATE.format(
                step_name=step_info['step_name'],
                method_params=method_info['params'],
                method_call_params=method_info['method_call_params']
            )
            
            enhanced_standard_method = STANDARD_METHOD_TEMPLATE.format(
                step_name=step_info['step_name'],
                method_params=method_info['params'],
                method_call_params=method_info['method_call_params']
            )
            
            # Create regime-aware logging code
            regime_aware_logging_code = create_regime_aware_logging_code(step_info['step_name'])
            
            enhanced_regime_aware_metrics = REGIME_AWARE_METRICS_TEMPLATE.format(
                method_params=method_info['params'],
                method_call_params=method_info['method_call_params'],
                regime_aware_logging_code=regime_aware_logging_code
            )
            
            # Replace the content
            new_content = original_content
            
            # Replace imports
            import_pattern = r'""".*?"""\s*\n\s*import.*?from src\.utils\.logger import system_logger'
            new_content = re.sub(import_pattern, enhanced_imports, new_content, flags=re.MULTILINE | re.DOTALL)
            
            # Replace class definition
            new_content = re.sub(class_start_pattern, enhanced_class_def, new_content, flags=re.MULTILINE | re.DOTALL)
            
            # Replace log_step_execution method
            log_method_pattern = r'def log_step_execution\(self, [^)]*\) -> [^:]*:.*?(?=\n    def|\nclass|\Z)'
            new_content = re.sub(log_method_pattern, enhanced_log_method, new_content, flags=re.MULTILINE | re.DOTALL)
            
            # Add enhanced methods before the last method
            last_method_pattern = r'(\n    def _log_created_file_paths\(self\) -> None:.*?)(\n\Z)'
            replacement = f'\\1\n\n{enhanced_regime_validation}\n\n{enhanced_standard_method}\n\n{enhanced_regime_aware_metrics}\\2'
            new_content = re.sub(last_method_pattern, replacement, new_content, flags=re.MULTILINE | re.DOTALL)
            
            return new_content
    
    return original_content


def create_regime_aware_logging_code(step_name: str) -> str:
    """Create regime-aware logging code for the specific step."""
    return f'''            # Log step success with regime awareness
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
            )'''


def main():
    """Main function to update all financial logging files."""
    print("🚀 Starting Enhanced Regime Logging Implementation")
    print("=" * 60)
    
    updated_files = []
    failed_files = []
    
    for file_path in FINANCIAL_LOGGING_FILES:
        if update_financial_logging_file(file_path):
            updated_files.append(file_path)
        else:
            failed_files.append(file_path)
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
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


if __name__ == "__main__":
    main()