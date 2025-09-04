#!/usr/bin/env python3
"""
Enhanced Backtesting Pipeline

This module provides a comprehensive, validated, and protected backtesting pipeline
with proper error handling, data validation, and operational safeguards.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import time
from dataclasses import dataclass

from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
    safe_file_exists,
    safe_json_load,
    safe_json_dump,
    ensure_directory,
)
from src.utils.compat import handle_errors

# Import our enhanced components
from .validation_framework import (
    BacktestingValidationOrchestrator,
    ValidationResult,
    ValidationStatus
)
from .step_validators import StepValidationOrchestrator
from .decorators import (
    BacktestingDecorators,
    DataFormattingDecorator,
    AnalysisProtectionDecorator,
    DataAccessProtectionDecorator,
    PerformanceMonitoringDecorator
)
from .common_utilities import (
    DataOperationUtilities,
    ErrorHandlingUtilities,
    PipelineManagementUtilities,
    ConfigurationUtilities,
    LoggingUtilities,
    OperationResult,
    OperationStatus
)


@dataclass
class BacktestingConfig:
    """Configuration for the enhanced backtesting pipeline."""
    symbol: str
    exchange: str
    data_dir: str = "data_cache"
    output_dir: str = "backtesting_results"
    log_dir: str = "logs/backtesting"
    
    # Data configuration
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timeframe: str = "1m"
    
    # Backtesting parameters
    initial_capital: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0005
    
    # Validation settings
    enable_validation: bool = True
    strict_mode: bool = True
    max_retries: int = 3
    
    # Performance settings
    max_workers: int = 4
    cache_results: bool = True
    profile_performance: bool = True


class EnhancedBacktestingPipeline:
    """Enhanced backtesting pipeline with comprehensive validation and error handling."""
    
    def __init__(self, config: BacktestingConfig):
        self.config = config
        self.logger = LoggingUtilities.setup_pipeline_logging(
            log_dir=config.log_dir,
            log_level="INFO"
        )
        
        # Initialize components
        self.validation_orchestrator = BacktestingValidationOrchestrator({})
        self.step_validator = StepValidationOrchestrator({})
        self.pipeline_manager = PipelineManagementUtilities(max_workers=config.max_workers)
        
        # Initialize results tracking
        self.pipeline_results = {}
        self.validation_results = {}
        
        self.logger.info(f"Enhanced backtesting pipeline initialized for {config.symbol} on {config.exchange}")
    
    @BacktestingDecorators.data_processing_pipeline(
        required_columns=["timestamp", "open", "high", "low", "close", "volume"],
        validate_price_data=True,
        handle_missing_data=True
    )
    @handle_errors(exceptions=(Exception,), default_return=None)
    async def load_and_validate_data(self) -> Optional[pd.DataFrame]:
        """Load and validate price data for backtesting."""
        operation_name = "data_loading"
        
        with ErrorHandlingUtilities.error_recovery_context(
            operation_name, self.config.symbol, self.config.exchange
        ) as context:
            
            LoggingUtilities.log_operation_start(
                self.logger, operation_name, self.config.symbol, self.config.exchange
            )
            
            # Construct data file path
            data_file = Path(self.config.data_dir) / f"aggtrades_{self.config.exchange}_{self.config.symbol}_consolidated.parquet"
            
            # Validate file access
            if self.config.enable_validation:
                file_validation = await self.validation_orchestrator.validate_pipeline_step(
                    "data_loading",
                    file_paths=[data_file],
                    symbol=self.config.symbol,
                    exchange=self.config.exchange
                )
                
                if file_validation.status == ValidationStatus.FAILED:
                    raise ValueError(f"Data file validation failed: {file_validation.message}")
            
            # Load data
            data = DataOperationUtilities.load_price_data(
                data_file,
                self.config.symbol,
                self.config.exchange,
                self.config.start_date,
                self.config.end_date
            )
            
            if data is None:
                raise ValueError("Failed to load price data")
            
            # Validate data quality
            if self.config.enable_validation:
                data_validation = await self.step_validator.validate_step(
                    "data_loading",
                    data=data,
                    symbol=self.config.symbol,
                    exchange=self.config.exchange
                )
                
                if data_validation.status == ValidationStatus.FAILED:
                    raise ValueError(f"Data quality validation failed: {data_validation.message}")
                
                self.validation_results["data_loading"] = data_validation
            
            # Validate data continuity
            continuity_stats = DataOperationUtilities.validate_data_continuity(
                data, expected_interval=self.config.timeframe
            )
            
            if not continuity_stats.get("valid", True):
                self.logger.warning(f"Data continuity issues: {continuity_stats}")
            
            LoggingUtilities.log_operation_end(
                self.logger, operation_name, self.config.symbol, self.config.exchange,
                success=True, additional_info={"data_shape": data.shape}
            )
            
            return data
    
    @BacktestingDecorators.analysis_operations(
        cache_results=True,
        prevent_lookahead=True,
        min_data_points=1000
    )
    @handle_errors(exceptions=(Exception,), default_return=None)
    async def engineer_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Engineer features for backtesting."""
        operation_name = "feature_engineering"
        
        with ErrorHandlingUtilities.error_recovery_context(
            operation_name, self.config.symbol, self.config.exchange
        ) as context:
            
            LoggingUtilities.log_operation_start(
                self.logger, operation_name, self.config.symbol, self.config.exchange
            )
            
            # Validate input data
            if self.config.enable_validation:
                input_validation = await self.step_validator.validate_step(
                    "feature_engineering",
                    data=data,
                    symbol=self.config.symbol,
                    exchange=self.config.exchange
                )
                
                if input_validation.status == ValidationStatus.FAILED:
                    raise ValueError(f"Feature engineering input validation failed: {input_validation.message}")
            
            # Create features (simplified example)
            features = data.copy()
            
            # Price-based features
            features["returns"] = features["close"].pct_change()
            features["log_returns"] = np.log(features["close"] / features["close"].shift(1))
            features["volatility"] = features["returns"].rolling(window=20).std()
            features["sma_20"] = features["close"].rolling(window=20).mean()
            features["sma_50"] = features["close"].rolling(window=50).mean()
            features["rsi"] = self._calculate_rsi(features["close"])
            
            # Volume-based features
            features["volume_sma"] = features["volume"].rolling(window=20).mean()
            features["volume_ratio"] = features["volume"] / features["volume_sma"]
            
            # Technical indicators
            features["bb_upper"], features["bb_lower"] = self._calculate_bollinger_bands(features["close"])
            features["macd"], features["macd_signal"] = self._calculate_macd(features["close"])
            
            # Remove rows with NaN values
            features = features.dropna()
            
            # Validate output features
            if self.config.enable_validation:
                output_validation = await self.step_validator.validate_step(
                    "feature_engineering",
                    features=features,
                    symbol=self.config.symbol,
                    exchange=self.config.exchange
                )
                
                if output_validation.status == ValidationStatus.FAILED:
                    raise ValueError(f"Feature engineering output validation failed: {output_validation.message}")
                
                self.validation_results["feature_engineering"] = output_validation
            
            LoggingUtilities.log_operation_end(
                self.logger, operation_name, self.config.symbol, self.config.exchange,
                success=True, additional_info={"feature_count": len(features.columns)}
            )
            
            return features
    
    @BacktestingDecorators.analysis_operations(
        cache_results=True,
        prevent_lookahead=True,
        min_data_points=100
    )
    @handle_errors(exceptions=(Exception,), default_return=None)
    async def train_model(self, features: pd.DataFrame) -> Optional[Any]:
        """Train a model for backtesting."""
        operation_name = "model_training"
        
        with ErrorHandlingUtilities.error_recovery_context(
            operation_name, self.config.symbol, self.config.exchange
        ) as context:
            
            LoggingUtilities.log_operation_start(
                self.logger, operation_name, self.config.symbol, self.config.exchange
            )
            
            # Prepare training data
            feature_columns = [col for col in features.columns if col not in ["timestamp", "open", "high", "low", "close", "volume"]]
            X = features[feature_columns]
            
            # Create simple target (price direction)
            y = (features["close"].shift(-1) > features["close"]).astype(int)
            
            # Remove last row (no target)
            X = X[:-1]
            y = y[:-1]
            
            # Validate training data
            if self.config.enable_validation:
                training_validation = await self.step_validator.validate_step(
                    "model_training",
                    X=X,
                    y=y,
                    symbol=self.config.symbol,
                    exchange=self.config.exchange
                )
                
                if training_validation.status == ValidationStatus.FAILED:
                    raise ValueError(f"Training data validation failed: {training_validation.message}")
            
            # Train model (simplified example using Random Forest)
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train model
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train)
                
                # Validate trained model
                if self.config.enable_validation:
                    model_validation = await self.step_validator.validate_step(
                        "model_training",
                        model=model,
                        symbol=self.config.symbol,
                        exchange=self.config.exchange
                    )
                    
                    if model_validation.status == ValidationStatus.FAILED:
                        raise ValueError(f"Trained model validation failed: {model_validation.message}")
                    
                    self.validation_results["model_training"] = model_validation
                
                LoggingUtilities.log_operation_end(
                    self.logger, operation_name, self.config.symbol, self.config.exchange,
                    success=True, additional_info={"model_type": type(model).__name__}
                )
                
                return model
                
            except ImportError:
                self.logger.warning("scikit-learn not available, using dummy model")
                # Return a dummy model for testing
                class DummyModel:
                    def predict(self, X):
                        return np.random.randint(0, 2, len(X))
                
                return DummyModel()
    
    @BacktestingDecorators.analysis_operations(
        cache_results=True,
        prevent_lookahead=True,
        min_data_points=100
    )
    @handle_errors(exceptions=(Exception,), default_return=None)
    async def run_backtest(self, features: pd.DataFrame, model: Any) -> Optional[Dict[str, Any]]:
        """Run backtesting with the trained model."""
        operation_name = "backtesting_execution"
        
        with ErrorHandlingUtilities.error_recovery_context(
            operation_name, self.config.symbol, self.config.exchange
        ) as context:
            
            LoggingUtilities.log_operation_start(
                self.logger, operation_name, self.config.symbol, self.config.exchange
            )
            
            # Prepare backtesting configuration
            backtest_config = {
                "initial_capital": self.config.initial_capital,
                "commission": self.config.commission,
                "slippage": self.config.slippage,
                "start_date": self.config.start_date,
                "end_date": self.config.end_date
            }
            
            # Validate backtest setup
            if self.config.enable_validation:
                setup_validation = await self.step_validator.validate_step(
                    "backtesting_execution",
                    config=backtest_config,
                    symbol=self.config.symbol,
                    exchange=self.config.exchange
                )
                
                if setup_validation.status == ValidationStatus.FAILED:
                    raise ValueError(f"Backtest setup validation failed: {setup_validation.message}")
            
            # Run backtesting simulation
            results = self._simulate_backtest(features, model, backtest_config)
            
            # Validate backtest results
            if self.config.enable_validation:
                results_validation = await self.step_validator.validate_step(
                    "backtesting_execution",
                    results=results,
                    symbol=self.config.symbol,
                    exchange=self.config.exchange
                )
                
                if results_validation.status == ValidationStatus.FAILED:
                    raise ValueError(f"Backtest results validation failed: {results_validation.message}")
                
                self.validation_results["backtesting_execution"] = results_validation
            
            LoggingUtilities.log_operation_end(
                self.logger, operation_name, self.config.symbol, self.config.exchange,
                success=True, additional_info={"total_return": results.get("total_return", 0)}
            )
            
            return results
    
    def _simulate_backtest(self, features: pd.DataFrame, model: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate backtesting with the trained model."""
        try:
            # Prepare features for prediction
            feature_columns = [col for col in features.columns if col not in ["timestamp", "open", "high", "low", "close", "volume"]]
            X = features[feature_columns]
            
            # Generate predictions
            predictions = model.predict(X)
            
            # Simulate trading
            capital = config["initial_capital"]
            position = 0
            trades = []
            
            for i in range(len(features)):
                if i == 0:
                    continue
                
                current_price = features.iloc[i]["close"]
                prediction = predictions[i-1] if i-1 < len(predictions) else 0
                
                # Simple trading logic
                if prediction == 1 and position <= 0:  # Buy signal
                    if position < 0:  # Close short position
                        pnl = (trades[-1]["price"] - current_price) * abs(position)
                        capital += pnl
                        trades.append({
                            "timestamp": features.iloc[i]["timestamp"],
                            "action": "close_short",
                            "price": current_price,
                            "quantity": abs(position),
                            "pnl": pnl
                        })
                    
                    # Open long position
                    quantity = capital * 0.95 / current_price  # Use 95% of capital
                    position = quantity
                    capital -= quantity * current_price * (1 + config["commission"])
                    trades.append({
                        "timestamp": features.iloc[i]["timestamp"],
                        "action": "buy",
                        "price": current_price,
                        "quantity": quantity,
                        "pnl": 0
                    })
                
                elif prediction == 0 and position > 0:  # Sell signal
                    # Close long position
                    pnl = (current_price - trades[-1]["price"]) * position
                    capital += pnl
                    trades.append({
                        "timestamp": features.iloc[i]["timestamp"],
                        "action": "sell",
                        "price": current_price,
                        "quantity": position,
                        "pnl": pnl
                    })
                    position = 0
            
            # Calculate final metrics
            if trades:
                total_trades = len([t for t in trades if t["action"] in ["sell", "close_short"]])
                winning_trades = len([t for t in trades if t["pnl"] > 0])
                total_pnl = sum(t["pnl"] for t in trades)
                total_return = total_pnl / config["initial_capital"]
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                # Calculate Sharpe ratio (simplified)
                returns = [t["pnl"] / config["initial_capital"] for t in trades if t["pnl"] != 0]
                sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
                
                # Calculate max drawdown (simplified)
                cumulative_returns = np.cumsum(returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = cumulative_returns - running_max
                max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            else:
                total_trades = 0
                win_rate = 0
                total_return = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "final_capital": capital,
                "trades": trades
            }
            
        except Exception as e:
            self.logger.exception(f"Error in backtest simulation: {e}")
            return {
                "total_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "total_trades": 0,
                "error": str(e)
            }
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    @handle_errors(exceptions=(Exception,), default_return=False)
    async def run_complete_pipeline(self) -> bool:
        """Run the complete enhanced backtesting pipeline."""
        pipeline_start_time = time.time()
        
        try:
            self.logger.info("=" * 80)
            self.logger.info("🚀 STARTING ENHANCED BACKTESTING PIPELINE")
            self.logger.info("=" * 80)
            self.logger.info(f"Symbol: {self.config.symbol}")
            self.logger.info(f"Exchange: {self.config.exchange}")
            self.logger.info(f"Data Directory: {self.config.data_dir}")
            self.logger.info(f"Output Directory: {self.config.output_dir}")
            self.logger.info(f"Validation Enabled: {self.config.enable_validation}")
            self.logger.info(f"Strict Mode: {self.config.strict_mode}")
            self.logger.info("=" * 80)
            
            # Step 1: Load and validate data
            self.logger.info("📊 STEP 1: Loading and validating data...")
            data = await self.load_and_validate_data()
            if data is None:
                raise ValueError("Failed to load and validate data")
            
            # Step 2: Engineer features
            self.logger.info("🔧 STEP 2: Engineering features...")
            features = await self.engineer_features(data)
            if features is None:
                raise ValueError("Failed to engineer features")
            
            # Step 3: Train model
            self.logger.info("🤖 STEP 3: Training model...")
            model = await self.train_model(features)
            if model is None:
                raise ValueError("Failed to train model")
            
            # Step 4: Run backtest
            self.logger.info("📈 STEP 4: Running backtest...")
            results = await self.run_backtest(features, model)
            if results is None:
                raise ValueError("Failed to run backtest")
            
            # Step 5: Save results
            self.logger.info("💾 STEP 5: Saving results...")
            await self._save_pipeline_results(results)
            
            # Generate final report
            pipeline_duration = time.time() - pipeline_start_time
            await self._generate_final_report(pipeline_duration)
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 ENHANCED BACKTESTING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"Total Return: {results.get('total_return', 0):.2%}")
            self.logger.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            self.logger.info(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
            self.logger.info(f"Win Rate: {results.get('win_rate', 0):.2%}")
            self.logger.info(f"Total Trades: {results.get('total_trades', 0)}")
            self.logger.info(f"Pipeline Duration: {pipeline_duration:.2f} seconds")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            pipeline_duration = time.time() - pipeline_start_time
            self.logger.exception(f"Enhanced backtesting pipeline failed: {e}")
            self.logger.error(f"Pipeline duration: {pipeline_duration:.2f} seconds")
            return False
    
    @handle_errors(exceptions=(Exception,), default_return=False)
    async def _save_pipeline_results(self, results: Dict[str, Any]) -> bool:
        """Save pipeline results to files."""
        try:
            output_dir = Path(self.config.output_dir)
            ensure_directory(output_dir)
            
            # Save main results
            results_file = output_dir / f"backtest_results_{self.config.symbol}_{self.config.exchange}_{format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')}.json"
            safe_json_dump(results, results_file, indent=2)
            
            # Save validation results
            if self.validation_results:
                validation_file = output_dir / f"validation_results_{self.config.symbol}_{self.config.exchange}_{format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')}.json"
                safe_json_dump(self.validation_results, validation_file, indent=2)
            
            # Save configuration
            config_file = output_dir / f"config_{self.config.symbol}_{self.config.exchange}_{format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')}.json"
            safe_json_dump(self.config.__dict__, config_file, indent=2)
            
            self.logger.info(f"Pipeline results saved to: {output_dir}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error saving pipeline results: {e}")
            return False
    
    @handle_errors(exceptions=(Exception,), default_return=False)
    async def _generate_final_report(self, pipeline_duration: float) -> bool:
        """Generate a comprehensive final report."""
        try:
            output_dir = Path(self.config.output_dir)
            ensure_directory(output_dir)
            
            report = {
                "pipeline_summary": {
                    "symbol": self.config.symbol,
                    "exchange": self.config.exchange,
                    "start_time": format_datetime(get_current_datetime()),
                    "duration_seconds": pipeline_duration,
                    "status": "COMPLETED"
                },
                "validation_summary": self.step_validator.get_validation_summary(),
                "configuration": self.config.__dict__,
                "results_summary": {
                    "total_validations": len(self.validation_results),
                    "validation_success_rate": sum(1 for v in self.validation_results.values() if v.status == ValidationStatus.PASSED) / len(self.validation_results) if self.validation_results else 0
                }
            }
            
            report_file = output_dir / f"pipeline_report_{self.config.symbol}_{self.config.exchange}_{format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')}.json"
            safe_json_dump(report, report_file, indent=2)
            
            self.logger.info(f"Final report saved to: {report_file}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error generating final report: {e}")
            return False


# Main execution function
async def run_enhanced_backtesting_pipeline(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE",
    config_overrides: Optional[Dict[str, Any]] = None
) -> bool:
    """Run the enhanced backtesting pipeline with the specified parameters."""
    
    # Create configuration
    config = BacktestingConfig(
        symbol=symbol,
        exchange=exchange
    )
    
    # Apply overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Create and run pipeline
    pipeline = EnhancedBacktestingPipeline(config)
    success = await pipeline.run_complete_pipeline()
    
    return success


if __name__ == "__main__":
    # Example usage
    asyncio.run(run_enhanced_backtesting_pipeline(
        symbol="ETHUSDT",
        exchange="BINANCE",
        config_overrides={
            "enable_validation": True,
            "strict_mode": True,
            "initial_capital": 10000.0
        }
    ))