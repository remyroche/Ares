# src/training/target_parameter_optimizer.py

import optuna
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression

from src.database.sqlite_manager import SQLiteManager
from src.utils.logger import system_logger

# Import the existing TechnicalAnalyzer to use its methods directly

# Suppress Optuna's informational messages for a cleaner log during optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TargetParameterOptimizer:
    """
    Optimizes Take Profit (TP), Stop Loss (SL), and Holding Period using Optuna.

    This class uses a simplified backtesting heuristic on a baseline ML model
    to find a synergistic combination of trade exit parameters before running
    the main, computationally expensive training.
    """

    def __init__(
        self,
        db_manager: SQLiteManager,
        symbol: str,
        timeframe: str,
        klines_data: pd.DataFrame,
        blank_training_mode: bool = False,
    ):
        """
        Initializes the optimizer.

        Args:
            db_manager (SQLiteManager): The database manager to fetch data (if needed for other features).
            symbol (str): The trading symbol (e.g., 'BTCUSDT').
            timeframe (str): The timeframe for the data (e.g., '1h').
            klines_data (pd.DataFrame): The klines data to use for optimization.
        """
        self.db_manager = (
            db_manager  # Retain for potential future use or other data sources
        )
        self.symbol = symbol
        self.timeframe = timeframe
        self.logger = system_logger.getChild("TargetParameterOptimizer")
        self.data = klines_data  # Use the passed klines_data directly
        self.signals = None
        self.blank_training_mode = blank_training_mode
        self._prepare_data_and_signals()

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds a comprehensive set of technical indicators for the baseline model."""
        self.logger.info("Adding technical indicators for baseline model...")
        try:
            # Ensure columns are named correctly for pandas_ta
            data.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
                errors="ignore",
            )

            # Log the actual column names after renaming
            self.logger.info(f"Data columns after renaming: {list(data.columns)}")
            
            # Check if required columns exist
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"Missing required columns: {missing_cols}")
                self.logger.info(f"Available columns: {list(data.columns)}")
                # Try to find alternative column names
                for col in missing_cols:
                    if col.lower() in data.columns:
                        data.rename(columns={col.lower(): col}, inplace=True)
                        self.logger.info(f"Renamed {col.lower()} to {col}")

            # Add technical indicators one by one with error handling
            indicators_added = []
            
            # RSI
            try:
                data.ta.rsi(length=14, append=True)
                indicators_added.append("RSI_14")
            except Exception as e:
                self.logger.warning(f"Failed to add RSI: {e}")
            
            # SMA
            try:
                data.ta.sma(length=50, append=True)
                indicators_added.append("SMA_50")
            except Exception as e:
                self.logger.warning(f"Failed to add SMA: {e}")
            
            # Bollinger Bands
            try:
                data.ta.bbands(length=20, append=True)
                indicators_added.extend(["BBU_20_2.0", "BBL_20_2.0"])
            except Exception as e:
                self.logger.warning(f"Failed to add Bollinger Bands: {e}")
            
            # MACD
            try:
                data.ta.macd(append=True)
                indicators_added.extend(["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"])
            except Exception as e:
                self.logger.warning(f"Failed to add MACD: {e}")
            
            # ATR
            try:
                data.ta.atr(length=14, append=True)
                indicators_added.append("ATRr_14")
            except Exception as e:
                self.logger.warning(f"Failed to add ATR: {e}")
            
            # ADX
            try:
                data.ta.adx(length=14, append=True)
                indicators_added.append("ADX_14")
            except Exception as e:
                self.logger.warning(f"Failed to add ADX: {e}")
            
            # OBV
            try:
                data.ta.obv(append=True)
                indicators_added.append("OBV")
            except Exception as e:
                self.logger.warning(f"Failed to add OBV: {e}")
            
            # VWAP
            try:
                data.ta.vwap(append=True)
                indicators_added.append("VWAP")
            except Exception as e:
                self.logger.warning(f"Failed to add VWAP: {e}")
            
            self.logger.info(f"Successfully added indicators: {indicators_added}")
            self.logger.info(f"Final data columns: {list(data.columns)}")
            
        except Exception as e:
            self.logger.error(f"Failed to add technical indicators: {e}", exc_info=True)
        return data

    def _prepare_data_and_signals(self):
        """
        Loads data, engineers features, and trains a simple model to get signals.
        """
        self.logger.info(
            "Preparing data and generating baseline signals for parameter optimization..."
        )

        if self.data.empty:
            raise ValueError("No klines data provided for TargetParameterOptimizer.")

        # Ensure timestamp is a datetime object and set as index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
            self.data.set_index("timestamp", inplace=True)

        # Add standard TAs
        self.data = self._add_technical_indicators(self.data.copy())

        # Use TechnicalAnalyzer for Order Book features (requires order book data, which is not passed here)
        # For this optimizer, we will skip order book features as they require live or pre-processed data.
        # This optimizer focuses on optimizing TP/SL based on klines and basic TAs.
        self.logger.info(
            "Skipping order book features for preliminary optimization (data not available)."
        )

        # Create a fixed target for the baseline model (e.g., price moves > 1% in next 5 bars)
        self.data["target"] = (
            self.data["Close"].shift(-5) > self.data["Close"] * 1.01
        ).astype(int)
        
        # Enhanced NaN analysis and logging
        self.logger.info(f"Data shape before handling NaN: {self.data.shape}")
        
        # Detailed NaN analysis
        nan_counts = self.data.isna().sum()
        total_rows = len(self.data)
        
        self.logger.info("🔍 DETAILED NaN ANALYSIS:")
        self.logger.info(f"   Total rows: {total_rows}")
        self.logger.info(f"   Columns with NaN values: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Analyze which columns have the most NaN values
        high_nan_cols = nan_counts[nan_counts > total_rows * 0.1]  # More than 10% NaN
        if not high_nan_cols.empty:
            self.logger.warning(f"   ⚠️  High NaN columns (>10%): {high_nan_cols.to_dict()}")
        
        # Check for columns that are entirely NaN
        all_nan_cols = nan_counts[nan_counts == total_rows]
        if not all_nan_cols.empty:
            self.logger.error(f"   ❌ All-NaN columns: {all_nan_cols.to_dict()}")
        
        # Check for rows with many NaN values
        row_nan_counts = self.data.isna().sum(axis=1)
        rows_with_many_nans = row_nan_counts[row_nan_counts > 5]  # More than 5 NaN values per row
        if not rows_with_many_nans.empty:
            self.logger.warning(f"   ⚠️  Rows with many NaN values: {len(rows_with_many_nans)} rows have >5 NaN values")
            self.logger.warning(f"   Max NaN values in a single row: {row_nan_counts.max()}")
        
        # Show sample of problematic rows
        if not rows_with_many_nans.empty:
            sample_problematic_rows = rows_with_many_nans.head(3)
            self.logger.info(f"   Sample problematic row indices: {sample_problematic_rows.index.tolist()}")
        
        # Additional data quality insights
        self.logger.info("📊 DATA QUALITY INSIGHTS:")
        self.logger.info(f"   Data types: {self.data.dtypes.to_dict()}")
        
        # Show value ranges for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.logger.info(f"   Numeric columns: {list(numeric_cols)}")
            for col in numeric_cols[:5]:  # Show first 5 numeric columns
                if not self.data[col].isna().all():
                    non_nan_vals = self.data[col].dropna()
                    if len(non_nan_vals) > 0:
                        self.logger.info(f"     {col}: min={non_nan_vals.min():.4f}, max={non_nan_vals.max():.4f}, mean={non_nan_vals.mean():.4f}")
        
        # Show unique values for categorical columns (first few)
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            self.logger.info(f"   Categorical columns: {list(categorical_cols)}")
            for col in categorical_cols[:3]:  # Show first 3 categorical columns
                unique_vals = self.data[col].dropna().unique()
                self.logger.info(f"     {col}: {len(unique_vals)} unique values, sample: {unique_vals[:3].tolist()}")
        
        if self.blank_training_mode:
            self.logger.info("🔧 BLANK TRAINING MODE: Replacing NaN values with random data instead of dropping rows")
            
            # Log replacement strategy for each column
            for column in self.data.columns:
                if self.data[column].isna().any():
                    nan_count = self.data[column].isna().sum()
                    non_nan_values = self.data[column].dropna()
                    
                    self.logger.info(f"   Processing column '{column}':")
                    self.logger.info(f"     - NaN count: {nan_count} ({nan_count/total_rows*100:.1f}%)")
                    self.logger.info(f"     - Data type: {self.data[column].dtype}")
                    self.logger.info(f"     - Non-NaN values available: {len(non_nan_values)}")
                    
                    if len(non_nan_values) > 0:
                        if non_nan_values.dtype in ['int64', 'float64']:
                            # For numeric columns, use random values from the existing distribution
                            random_values = np.random.choice(non_nan_values, size=nan_count)
                            self.data.loc[self.data[column].isna(), column] = random_values
                            self.logger.info(f"     - Strategy: Random sampling from existing distribution")
                            self.logger.info(f"     - Sample replacement values: {random_values[:3].tolist()}")
                        else:
                            # For non-numeric columns, use the most common value
                            most_common = non_nan_values.mode().iloc[0] if len(non_nan_values.mode()) > 0 else 0
                            self.data[column].fillna(most_common, inplace=True)
                            self.logger.info(f"     - Strategy: Fill with most common value")
                            self.logger.info(f"     - Most common value: {most_common}")
                    else:
                        # If no non-NaN values exist, fill with 0
                        self.data[column].fillna(0, inplace=True)
                        self.logger.warning(f"     - Strategy: Fill with 0 (no valid data available)")
                        self.logger.warning(f"     - WARNING: Column '{column}' had no valid data!")
            
            self.logger.info(f"Data shape after NaN replacement: {self.data.shape}")
            
            # Verify no NaN values remain
            remaining_nans = self.data.isna().sum().sum()
            if remaining_nans > 0:
                self.logger.error(f"❌ ERROR: {remaining_nans} NaN values still remain after replacement!")
            else:
                self.logger.info(f"✅ All NaN values successfully replaced")
                
        else:
            # Normal mode: drop NaN values
            self.logger.info("🔧 NORMAL MODE: Dropping rows with NaN values")
            self.data.dropna(inplace=True)
            self.logger.info(f"Data shape after dropping NaN: {self.data.shape}")
            self.logger.info(f"Rows dropped: {total_rows - len(self.data)} ({((total_rows - len(self.data))/total_rows)*100:.1f}%)")

        # Define features for the baseline model
        features = [
            "RSI_14",
            "SMA_50",
            "BBU_20_2.0",
            "BBL_20_2.0",
            "MACD_12_26_9",
            "MACDs_12_26_9",
            "MACDh_12_26_9",
            "ATRr_14",
            "ADX_14",
            "OBV",
            "VWAP",
            # 'bid_ask_spread', 'order_book_imbalance' # Removed as order book data is not passed
        ]

        features_in_data = [f for f in features if f in self.data.columns]
        self.logger.info(f"Available features in data: {features_in_data}")
        self.logger.info(f"Missing features: {[f for f in features if f not in self.data.columns]}")
        
        if not features_in_data:
            # Try to find alternative feature names
            all_columns = list(self.data.columns)
            self.logger.info(f"All available columns: {all_columns}")
            
            # Look for any technical indicator columns
            ta_columns = [col for col in all_columns if any(indicator in col for indicator in ['RSI', 'SMA', 'BB', 'MACD', 'ATR', 'ADX', 'OBV', 'VWAP'])]
            if ta_columns:
                self.logger.info(f"Found technical indicator columns: {ta_columns}")
                features_in_data = ta_columns[:5]  # Use first 5 available indicators
                self.logger.info(f"Using available indicators: {features_in_data}")
            else:
                raise ValueError("No features available for baseline model training. No technical indicators found in data.")

        X = self.data[features_in_data]
        y = self.data["target"]

        # Check for NaN values in features and target
        self.logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        self.logger.info(f"X NaN count: {X.isna().sum().to_dict()}")
        self.logger.info(f"y NaN count: {y.isna().sum()}")
        
        # Remove rows with NaN values in features or target
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.logger.info(f"After removing NaN - X shape: {X.shape}, y shape: {y.shape}")

        if X.empty:
            # Try to use a minimal set of features if all features are empty
            self.logger.warning("Feature set is empty after data cleaning. Trying fallback approach...")
            
            # Check if we have any numeric columns that could be used as features
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.logger.info(f"Available numeric columns: {numeric_cols}")
            
            # Remove target and signal columns from potential features
            potential_features = [col for col in numeric_cols if col not in ['target', 'signal']]
            
            if len(potential_features) >= 3:
                self.logger.info(f"Using fallback features: {potential_features[:5]}")
                X = self.data[potential_features[:5]]
                y = self.data["target"]
                
                # Remove rows with NaN values
                valid_mask = ~(X.isna().any(axis=1) | y.isna())
                X = X[valid_mask]
                y = y[valid_mask]
                
                if X.empty:
                    raise ValueError("Feature set is empty even with fallback approach. Check data quality.")
                else:
                    self.logger.info(f"Fallback successful. Using {len(potential_features[:5])} features with {len(X)} samples")
            else:
                raise ValueError("Feature set is empty, cannot train baseline model. Insufficient numeric data available.")

        self.logger.info(f"Training baseline model with {len(features_in_data)} features: {features_in_data}")
        self.logger.info(f"Final X shape: {X.shape}, y shape: {y.shape}")
        self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        model = LogisticRegression(
            solver="liblinear", random_state=42, class_weight="balanced", max_iter=1000
        )
        model.fit(X, y)

        self.signals = pd.Series(model.predict(X), index=X.index)
        self.data["signal"] = self.signals

        self.logger.info(
            f"Baseline signals generated. Found {len(self.signals[self.signals == 1])} potential long signals."
        )

    def _run_backtest(
        self, tp_threshold: float, sl_threshold: float, holding_period: int
    ) -> pd.Series:
        """
        Runs a simplified backtest with TP, SL, and a time-based exit (holding period).
        """
        pnls = []
        position_open = False
        entry_price = 0.0
        bars_held = 0

        close_prices = self.data["Close"].to_numpy()
        low_prices = self.data["Low"].to_numpy()
        high_prices = self.data["High"].to_numpy()
        signals = self.data["signal"].to_numpy()

        for i in range(1, len(self.data)):
            if (
                not position_open and signals[i - 1] == 1
            ):  # Enter on signal from previous bar
                position_open = True
                entry_price = close_prices[i - 1]
                bars_held = 0

            if position_open:
                bars_held += 1
                if high_prices[i] >= entry_price * (1 + tp_threshold):
                    pnls.append(tp_threshold)
                    position_open = False
                elif low_prices[i] <= entry_price * (1 - sl_threshold):
                    pnls.append(-sl_threshold)
                    position_open = False
                elif bars_held >= holding_period:
                    pnl = (close_prices[i] - entry_price) / entry_price
                    pnls.append(pnl)
                    position_open = False

        return pd.Series(pnls)

    def objective(self, trial: optuna.trial.Trial) -> float:
        """The objective function for Optuna to maximize."""
        tp = trial.suggest_float("tp_threshold", 0.002, 0.05, log=True)
        sl = trial.suggest_float("sl_threshold", 0.002, 0.05, log=True)
        holding_period = trial.suggest_int("holding_period", 5, 50)

        pnl_series = self._run_backtest(
            tp_threshold=tp, sl_threshold=sl, holding_period=holding_period
        )

        if len(pnl_series) < 20:  # Ensure enough trades for a meaningful result
            return -1.0

        std_dev = pnl_series.std()
        if std_dev < 1e-9:
            return 0.0

        sharpe_ratio = pnl_series.mean() / std_dev
        return sharpe_ratio if pd.notna(sharpe_ratio) else -1.0

    def run_optimization(self, n_trials: int = 200) -> dict:
        """Executes the Optuna optimization study."""
        self.logger.info(
            f"Starting target parameter optimization for {n_trials} trials..."
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        if not study.best_trial or study.best_value <= 0:
            self.logger.warning(
                "Optuna study did not find a profitable set of parameters. Falling back to defaults."
            )
            return {"tp_threshold": 0.01, "sl_threshold": 0.01, "holding_period": 24}

        self.logger.info("Target parameter optimization finished.")
        self.logger.info(f"Best trial value (Sharpe Ratio): {study.best_value:.4f}")
        self.logger.info(f"Best params: {study.best_params}")

        return study.best_params
