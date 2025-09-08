from ..standardized_parquet_handler import standardized_parquet_handler
"""
Financial metrics logging for Step18_Financial.
Independent logging module that can be used without the reporting system.

Enhanced with per-HMM regime logging and fail-fast validation.
"""

import pandas as pd

from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum
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

logger = system_logger.getChild('Step18Financiallogging')

class ValidationStatus(Enum):
    """Validation status enumeration."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    FAILED = "failed"

@dataclass
class ValidationResult:
    """Validation result data class."""
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None

def _validate_financial_logger_inputs(symbol: str, exchange: str, timeframe: str, enable_enhanced_logging: bool) -> ValidationResult:
    """Validate inputs for financial logger initialization."""
    try:
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            return ValidationResult(ValidationStatus.ERROR, "Symbol must be a non-empty string")
        
        if not symbol.isupper() or len(symbol) < 3:
            return ValidationResult(ValidationStatus.ERROR, f"Invalid symbol format: {symbol}")
        
        # Validate exchange
        valid_exchanges = {'BINANCE', 'MEXC', 'GATEIO', 'KUCOIN', 'OKX'}
        if exchange not in valid_exchanges:
            return ValidationResult(ValidationStatus.ERROR, f"Invalid exchange: {exchange}")
        
        # Validate timeframe
        valid_timeframes = {'1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'}
        if timeframe not in valid_timeframes:
            return ValidationResult(ValidationStatus.ERROR, f"Invalid timeframe: {timeframe}")
        
        # Validate enhanced logging flag
        if not isinstance(enable_enhanced_logging, bool):
            return ValidationResult(ValidationStatus.ERROR, "enable_enhanced_logging must be a boolean")
        
        return ValidationResult(ValidationStatus.SUCCESS, "All inputs valid")
        
    except Exception as e:
        return ValidationResult(ValidationStatus.ERROR, f"Validation error: {e}")

def _validate_dataframe_input(data: Optional[pd.DataFrame], required_columns: Optional[List[str]] = None) -> ValidationResult:
    """Validate DataFrame input for financial logging."""
    try:
        if data is None:
            return ValidationResult(ValidationStatus.WARNING, "No data provided, using fallback logging")
        
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(ValidationStatus.ERROR, "Data must be a pandas DataFrame")
        
        if data.empty:
            return ValidationResult(ValidationStatus.WARNING, "Empty DataFrame provided")
        
        # Check required columns if specified
        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return ValidationResult(ValidationStatus.ERROR, f"Missing required columns: {missing_columns}")
        
        return ValidationResult(ValidationStatus.SUCCESS, "DataFrame validation passed")
        
    except Exception as e:
        return ValidationResult(ValidationStatus.ERROR, f"DataFrame validation error: {e}")

class Step18FinancialloggingFinancialLogger:
    """Independent financial metrics logger for Step18_Financial with enhanced regime logging."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str, enable_enhanced_logging: bool = True):
        # Validate inputs first
        validation_result = _validate_financial_logger_inputs(symbol, exchange, timeframe, enable_enhanced_logging)
        if validation_result.status == ValidationStatus.ERROR:
            raise ValueError(f"Invalid inputs: {validation_result.message}")
        elif validation_result.status == ValidationStatus.WARNING:
            logger.warning(f"Input validation warning: {validation_result.message}")
        
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.enable_enhanced_logging = enable_enhanced_logging
        
        try:
            # Use smart logger that automatically chooses enhanced or base logger
            self.financial_logger = get_smart_financial_metrics_logger(use_enhanced=enable_enhanced_logging)
            
            # Store enhanced logger separately if available
            if ENHANCED_LOGGING_AVAILABLE and enable_enhanced_logging:
                self.enhanced_logger = get_enhanced_financial_metrics_logger()
            else:
                self.enhanced_logger = None
                
            logger.info(f"✅ Financial logger initialized for {symbol}/{exchange}/{timeframe}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize financial logger: {e}")
            # Fallback to basic logger
            self.financial_logger = None
            self.enhanced_logger = None
    
    async def log_step_execution(self, *args, data: Optional[pd.DataFrame] = None, **kwargs) -> bool:
        """
        Log comprehensive financial metrics for Step18_Financial execution with enhanced regime validation.
        
        Args:
            *args: Step execution arguments
            data: DataFrame for regime validation (optional)
            **kwargs: Additional keyword arguments
            
        Returns:
            True if logging succeeded, False if fail-fast conditions triggered
        """
        try:
            # Validate DataFrame input if provided
            if data is not None:
                data_validation = _validate_dataframe_input(data, required_columns=['composite_cluster_id'])
                if data_validation.status == ValidationStatus.ERROR:
                    logger.error(f"❌ Data validation failed: {data_validation.message}")
                    return False
                elif data_validation.status == ValidationStatus.WARNING:
                    logger.warning(f"⚠️ Data validation warning: {data_validation.message}")
            
            # Validate financial logger is available
            if self.financial_logger is None:
                logger.error("❌ Financial logger not initialized")
                return False
            
            # Use enhanced logging if available and data is provided
            if self.enhanced_logger and data is not None:
                return await self._log_with_enhanced_regime_validation(*args, data=data, **kwargs)
            else:
                # Fallback to standard logging
                return await self._log_with_standard_method(*args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to log financial metrics: {e}")
            return False
    
    async def _log_with_enhanced_regime_validation(self, *args, data: pd.DataFrame, **kwargs) -> bool:
        """Log with enhanced regime validation and fail-fast checks."""
        try:
            # Validate regime data first
            if validate_and_log_regime_data:
                validation_success = validate_and_log_regime_data(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step18_Financial",
                    data=data,
                    regime_column='composite_cluster_id'
                )
                
                if not validation_success:
                    logger.error("🚨 Regime validation failed for Step18_Financial")
                    return False
            
            # Log step start
            self.financial_logger.log_step_start("Step18_Financial", self.symbol, self.exchange, self.timeframe)
            
            # Log all financial metrics with regime awareness
            success = await self._log_financial_metrics_with_regime_awareness(*args, data=data, **kwargs)
            
            # Log file paths
            self._log_created_file_paths()
            
            # Log step end
            self.financial_logger.log_step_end(
                "Step18_Financial", 
                self.symbol, 
                self.exchange, 
                self.timeframe, 
                success=success
            )
            
            return success
            
        except Exception as e:
            self.financial_logger.log_step_end(
                "Step18_Financial", 
                self.symbol, 
                self.exchange, 
                self.timeframe, 
                success=False, 
                error_message=str(e)
            )
            logger.error(f"Enhanced regime validation logging failed: {e}")
            return False
    
    async def _log_with_standard_method(self, *args, **kwargs) -> bool:
        """Log using standard method (fallback)."""
        with financial_metrics_context(
            step_name="Step18_Financial",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step18_Financial", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                await self._log_financial_metrics_from_results(*args, **kwargs)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step18_Financial", self.symbol, self.exchange, self.timeframe, success=True)
                
                return True
                
            except Exception as e:
                self.financial_logger.log_step_end("Step18_Financial", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
                return False
    
    async def _log_financial_metrics_with_regime_awareness(self, *args, data: pd.DataFrame, **kwargs) -> bool:
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
                step_name="Step18_Financial",
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
                step_name="Step18_Financial",
                data=data
            )
            
            # Log regime-specific metrics if enhanced logger is available
            if self.enhanced_logger and data is not None and 'composite_cluster_id' in data.columns:
                regime_data = data['composite_cluster_id'].dropna()
                regime_counts = regime_data.value_counts()
                
                regime_metrics = {}
                for regime_id, count in regime_counts.items():
                    regime_metrics[str(regime_id)] = {
                        'sample_count': float(count),
                        'regime_processed': 1.0
                    }
                
                # Use enhanced logger for per-regime metrics
                success &= self.enhanced_logger.log_per_regime_metrics(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step18_Financial",
                    regime_metrics=regime_metrics,
                    data=data
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to log financial metrics with regime awareness: {e}")
            return False
    
    async def _log_financial_metrics_from_results(self, *args, **kwargs) -> None:
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
                step_name="Step18_Financial"
            )
        except Exception as e:
            logger.error(f"Failed to log financial metrics from results: {e}")
    
    def _log_created_file_paths(self) -> None:
        """Log file paths that were created during this step."""
        try:
            if hasattr(self.financial_logger, 'current_file_path') and self.financial_logger.current_file_path:
                logger.info(f"📁 Financial metrics file created: {self.financial_logger.current_file_path}")
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="metrics_file_path",
                    metric_value=0.0,
                    metric_type="file_path",
                    step_name="Step18_Financial",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step18_Financial")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")

# Enhanced Step18Financiallogging Financial Logger with Regime-Aware Decorator Support
class EnhancedStep18FinancialloggingFinancialLogger(Step18FinancialloggingFinancialLogger):
    """Enhanced Step18Financiallogging Financial Logger with automatic regime-aware logging decorator support."""
    
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
