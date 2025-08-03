# src/training/target_parameter_optimizer.py

import os

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.database.sqlite_manager import SQLiteManager
from src.utils.error_handler import handle_errors, handle_specific_errors
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
        self.db_manager = db_manager
        self.symbol = symbol
        self.timeframe = timeframe
        self.klines_data = klines_data.copy()  # Store original data for fallback
        self.data = klines_data.copy()
        self.blank_training_mode = blank_training_mode
        self.logger = system_logger.getChild("TargetParameterOptimizer")
        self.signals = None

        # Prepare data and signals
        self._prepare_data_and_signals()

    @handle_specific_errors(
        error_handlers={
            ImportError: (None, "Failed to import TechnicalAnalyzer"),
            AttributeError: (None, "TechnicalAnalyzer method not found"),
            ValueError: (None, "Invalid data format for TechnicalAnalyzer"),
        },
        default_return=None,
        context="technical indicators",
    )
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds a comprehensive set of technical indicators for the baseline model."""
        self.logger.info("Adding technical indicators for baseline model...")

        # Create a copy to work with
        result_data = data.copy()

        # Prepare data for TechnicalAnalyzer
        analysis_data = self._prepare_data_for_analyzer(data)

        # Get indicators from TechnicalAnalyzer
        indicators = self._get_technical_indicators(analysis_data)

        # Add indicators to result data
        indicators_added = self._add_indicators_to_data(result_data, indicators)

        self.logger.info(f"Successfully added indicators: {indicators_added}")
        return result_data

    def _prepare_data_for_analyzer(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for TechnicalAnalyzer by renaming columns."""
        analysis_data = data.copy()

        # Rename columns to match what TechnicalAnalyzer expects
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }

        # Only rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in analysis_data.columns:
                analysis_data[new_name] = analysis_data[old_name]

        # Add timestamp column if not present
        if (
            "timestamp" not in analysis_data.columns
            and analysis_data.index.name == "timestamp"
        ):
            analysis_data["timestamp"] = analysis_data.index

        return analysis_data

    @handle_errors(
        exceptions=(ImportError, AttributeError, ValueError),
        default_return=None,
        context="TechnicalAnalyzer",
    )
    def _get_technical_indicators(self, analysis_data: pd.DataFrame) -> dict:
        """Get technical indicators using TechnicalAnalyzer."""
        from src.analyst.technical_analyzer import TechnicalAnalyzer

        analyzer = TechnicalAnalyzer()
        return analyzer.analyze(analysis_data)

    def _add_indicators_to_data(
        self,
        result_data: pd.DataFrame,
        indicators: dict,
    ) -> list:
        """Add indicators to the result data and return list of added indicators."""
        indicators_added = []

        if not indicators:
            return indicators_added

        # Add RSI
        if "rsi" in indicators and indicators["rsi"] is not None:
            result_data["RSI_14"] = indicators["rsi"]
            indicators_added.append("RSI_14")

        # Add MACD components
        self._add_macd_indicators(result_data, indicators, indicators_added)

        # Add Moving Averages
        self._add_moving_averages(result_data, indicators, indicators_added)

        # Add VWAP
        if "vwap" in indicators and indicators["vwap"] is not None:
            result_data["VWAP"] = indicators["vwap"]
            indicators_added.append("VWAP")

        # Add other indicators
        self._add_other_indicators(result_data, indicators, indicators_added)

        return indicators_added

    def _add_macd_indicators(
        self,
        result_data: pd.DataFrame,
        indicators: dict,
        indicators_added: list,
    ):
        """Add MACD indicators to result data."""
        if "macd" in indicators and isinstance(indicators["macd"], dict):
            macd_data = indicators["macd"]
            if "macd" in macd_data:
                result_data["MACD_12_26_9"] = macd_data["macd"]
                indicators_added.append("MACD_12_26_9")
            if "signal" in macd_data:
                result_data["MACDs_12_26_9"] = macd_data["signal"]
                indicators_added.append("MACDs_12_26_9")
            if "histogram" in macd_data:
                result_data["MACDh_12_26_9"] = macd_data["histogram"]
                indicators_added.append("MACDh_12_26_9")

    def _add_moving_averages(
        self,
        result_data: pd.DataFrame,
        indicators: dict,
        indicators_added: list,
    ):
        """Add moving averages to result data."""
        if "moving_averages" in indicators and isinstance(
            indicators["moving_averages"],
            dict,
        ):
            mas = indicators["moving_averages"]
            if "sma_50" in mas:
                result_data["SMA_50"] = mas["sma_50"]
                indicators_added.append("SMA_50")

    def _add_other_indicators(
        self,
        result_data: pd.DataFrame,
        indicators: dict,
        indicators_added: list,
    ):
        """Add other technical indicators to result data."""
        if "williams_r" in indicators and indicators["williams_r"] is not None:
            result_data["WILLR_14"] = indicators["williams_r"]
            indicators_added.append("WILLR_14")

        if "cci" in indicators and indicators["cci"] is not None:
            result_data["CCI_20"] = indicators["cci"]
            indicators_added.append("CCI_20")

        # Add Bollinger Bands
        if "bollinger_bands" in indicators and isinstance(
            indicators["bollinger_bands"],
            dict,
        ):
            bb_data = indicators["bollinger_bands"]
            if "BBU_20_2.0" in bb_data:
                result_data["BBU_20_2.0"] = bb_data["BBU_20_2.0"]
                indicators_added.append("BBU_20_2.0")
            if "BBL_20_2.0" in bb_data:
                result_data["BBL_20_2.0"] = bb_data["BBL_20_2.0"]
                indicators_added.append("BBL_20_2.0")

        # Add ATR
        if "atr" in indicators and indicators["atr"] is not None:
            result_data["ATR_14"] = indicators["atr"]
            indicators_added.append("ATR_14")

        # Add ATR Ratio
        if "atr_ratio" in indicators and indicators["atr_ratio"] is not None:
            result_data["ATRr_14"] = indicators["atr_ratio"]
            indicators_added.append("ATRr_14")

        # Add ADX
        if "adx" in indicators and indicators["adx"] is not None:
            result_data["ADX_14"] = indicators["adx"]
            indicators_added.append("ADX_14")

        # Add OBV
        if "obv" in indicators and indicators["obv"] is not None:
            result_data["OBV"] = indicators["obv"]
            indicators_added.append("OBV")

    @handle_errors(
        exceptions=(ImportError, AttributeError, ValueError),
        default_return=None,
        context="pandas_ta fallback",
    )
    def _add_fallback_indicators(self, result_data: pd.DataFrame) -> list:
        """Add basic indicators using pandas_ta as fallback."""
        import pandas_ta as ta

        indicators_added = []

        # Ensure we have the required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [col for col in required_cols if col in result_data.columns]

        if len(available_cols) >= 4:  # Need at least OHLC
            # Add basic indicators
            if "close" in result_data.columns:
                # RSI
                rsi = ta.rsi(result_data["close"], length=14)
                if rsi is not None and not rsi.isna().all():
                    result_data["RSI_14"] = rsi
                    indicators_added.append("RSI_14")

                # SMA
                sma = ta.sma(result_data["close"], length=50)
                if sma is not None and not sma.isna().all():
                    result_data["SMA_50"] = sma
                    indicators_added.append("SMA_50")

        return indicators_added

    def _prepare_data_and_signals(self):
        """Prepare data structure and create signals for optimization."""
        self.logger.info("Preparing data structure and signals...")

        # Prepare data structure
        self._prepare_data_structure()

        # Add technical indicators
        self._add_technical_indicators_to_data()

        # Create target variable
        self._create_target_variable()

        # Analyze and handle data quality
        self._analyze_and_handle_data_quality()

        # Prepare features and train baseline model
        self._prepare_features_and_train_model()

        self.logger.info("Data preparation completed successfully")

    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=None,
        context="data structure preparation",
    )
    def _prepare_data_structure(self):
        """Prepare the data structure with proper timestamp indexing."""
        self.logger.info("Preparing data structure...")
        self.logger.info(f"Available columns: {list(self.data.columns)}")

        # Try to use open_time column first
        if self._try_use_open_time_column():
            return

        # Try to use timestamp column
        if self._try_use_timestamp_column():
            return

        # Try to use index if it's already a datetime
        if self._try_use_datetime_index():
            return

        # Create simple index as fallback
        self._create_simple_index()

    def _try_use_open_time_column(self) -> bool:
        """Try to use open_time column for timestamp indexing."""
        if "open_time" not in self.data.columns:
            return False

        self.logger.info("Found 'open_time' column, checking if it's usable")

        # Check if open_time column has valid data
        valid_open_time = self.data["open_time"].notna()
        if not valid_open_time.any():
            self.logger.warning("open_time column has no valid data")
            return False

        self.logger.info(
            f"Using open_time column with {valid_open_time.sum()} valid timestamps",
        )

        try:
            # Convert open_time to datetime
            self.data["timestamp"] = pd.to_datetime(
                self.data["open_time"],
                errors="coerce",
            )

            # Remove rows with invalid timestamps
            invalid_timestamps = self.data["timestamp"].isna()
            if invalid_timestamps.any():
                self.logger.warning(
                    f"Removing {invalid_timestamps.sum()} rows with invalid timestamps",
                )
                self.data = self.data[~invalid_timestamps]

            self.data.set_index("timestamp", inplace=True)
            return True

        except Exception as e:
            self.logger.warning(f"Failed to parse open_time: {e}")
            return False

    def _try_use_timestamp_column(self) -> bool:
        """Try to use timestamp column for indexing."""
        if "timestamp" not in self.data.columns:
            return False

        self.logger.info("Found 'timestamp' column, using it as index")

        try:
            # Handle timestamp parsing with mixed formats
            self.data["timestamp"] = pd.to_datetime(
                self.data["timestamp"],
                format="mixed",
            )
        except ValueError:
            try:
                # Try with ISO8601 format
                self.data["timestamp"] = pd.to_datetime(
                    self.data["timestamp"],
                    format="ISO8601",
                )
            except ValueError:
                try:
                    # Try with errors='coerce' to handle problematic timestamps
                    self.data["timestamp"] = pd.to_datetime(
                        self.data["timestamp"],
                        errors="coerce",
                    )

                    # Remove rows with invalid timestamps
                    invalid_timestamps = self.data["timestamp"].isna()
                    if invalid_timestamps.any():
                        self.logger.warning(
                            f"Removing {invalid_timestamps.sum()} rows with invalid timestamps",
                        )
                        self.data = self.data[~invalid_timestamps]
                except Exception as e:
                    self.logger.error(f"Failed to parse timestamps: {e}")
                    return False

        self.data.set_index("timestamp", inplace=True)
        return True

    def _try_use_datetime_index(self) -> bool:
        """Try to use existing datetime index."""
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.logger.info("Data already has datetime index, using it")
            return True
        return False

    def _create_simple_index(self):
        """Create a simple timestamp index as fallback."""
        self.logger.warning("No valid timestamp column found, creating simple index")
        self.data["timestamp"] = pd.date_range(
            start="2025-01-01",
            periods=len(self.data),
            freq="1H",
        )
        self.data.set_index("timestamp", inplace=True)

    def _add_technical_indicators_to_data(self):
        """Add technical indicators to the data."""
        # Add standard TAs
        self.data = self._add_technical_indicators(self.data.copy())

        # Add real order book features from cached data
        self._add_real_order_book_features()

    def _add_real_order_book_features(self):
        """Add real order book features from cached data."""
        self.logger.info("Loading real order book features from cached data...")

        try:
            # Load aggregated trades data for the current exchange and symbol
            # Note: This is a simplified approach - in a full implementation, 
            # the exchange and symbol should be passed as parameters
            agg_trades_files = [
                f
                for f in os.listdir("data_cache")
                if f.startswith("aggtrades_") and f.endswith(".csv")
            ]

            if not agg_trades_files:
                self.logger.warning("No aggregated trades files found in data_cache")
                self._add_mock_order_book_features()
                return

            # Load the most recent aggregated trades file
            latest_file = sorted(agg_trades_files)[-1]
            agg_trades_path = os.path.join("data_cache", latest_file)

            self.logger.info(f"Loading aggregated trades from: {latest_file}")
            agg_trades_df = pd.read_csv(agg_trades_path)
            agg_trades_df["timestamp"] = pd.to_datetime(agg_trades_df["timestamp"])
            agg_trades_df.set_index("timestamp", inplace=True)

            # Calculate bid-ask spread from aggregated trades
            # Group by time windows and calculate spread metrics
            agg_trades_df["time_window"] = agg_trades_df.index.floor("1H")

            # Calculate spread metrics for each hour
            spread_metrics = (
                agg_trades_df.groupby("time_window")
                .agg(
                    {
                        "price": ["min", "max", "std"],
                        "quantity": "sum",
                        "is_buyer_maker": "sum",
                    },
                )
                .round(4)
            )

            # Flatten column names
            spread_metrics.columns = [
                "price_min",
                "price_max",
                "price_std",
                "total_quantity",
                "buyer_maker_count",
            ]
            spread_metrics["bid_ask_spread"] = (
                spread_metrics["price_max"] - spread_metrics["price_min"]
            ) / spread_metrics["price_min"]

            # Calculate order book imbalance (ratio of buyer vs seller initiated trades)
            spread_metrics["total_trades"] = agg_trades_df.groupby("time_window").size()
            spread_metrics["order_book_imbalance"] = (
                spread_metrics["buyer_maker_count"] / spread_metrics["total_trades"]
            )

            # Resample to match the main data timeframe
            if "timestamp" in self.data.columns:
                self.data.set_index("timestamp", inplace=True)

            # Merge order book features with main data
            self.data = self.data.join(
                spread_metrics[["bid_ask_spread", "order_book_imbalance"]],
                how="left",
            )

            # Fill missing values with forward fill then backward fill
            self.data["bid_ask_spread"] = (
                self.data["bid_ask_spread"]
                .ffill()
                .bfill()
            )
            self.data["order_book_imbalance"] = (
                self.data["order_book_imbalance"]
                .ffill()
                .bfill()
            )

            # Fill any remaining NaN values with reasonable defaults
            self.data["bid_ask_spread"] = self.data["bid_ask_spread"].fillna(
                0.001,
            )  # 0.1% default spread
            self.data["order_book_imbalance"] = self.data[
                "order_book_imbalance"
            ].fillna(0.5)  # 50/50 default

            self.logger.info("Real order book features loaded successfully")

        except Exception as e:
            self.logger.warning(f"Failed to load real order book features: {e}")
            self.logger.info("Falling back to mock order book features")
            self._add_mock_order_book_features()

    def _add_mock_order_book_features(self):
        """Add mock order book features for preliminary optimization."""
        self.logger.info(
            "Creating mock order book features for preliminary optimization...",
        )

        try:
            # Create mock bid-ask spread based on price volatility
            if "close" in self.data.columns:
                # Calculate price volatility as a proxy for spread
                price_volatility = self.data["close"].rolling(window=20).std()
                mock_spread = price_volatility * 0.001  # 0.1% of volatility
                self.data["bid_ask_spread"] = mock_spread.fillna(mock_spread.mean())

                # Create mock order book imbalance based on price momentum
                price_momentum = self.data["close"].pct_change(5)
                mock_imbalance = np.where(
                    price_momentum > 0,
                    0.6,
                    0.4,
                )  # 60/40 split based on momentum
                self.data["order_book_imbalance"] = mock_imbalance

                self.logger.info("Mock order book features created successfully")
            else:
                self.logger.warning(
                    "Cannot create mock order book features - no 'close' column available",
                )

        except Exception as e:
            self.logger.warning(f"Failed to create mock order book features: {e}")

    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=None,
        context="target variable creation",
    )
    def _create_target_variable(self):
        """Create the target variable for the baseline model."""
        # Create a fixed target for the baseline model (e.g., price moves > 1% in next 5 bars)
        # Use lowercase column names since that's what the data has
        close_col = "close" if "close" in self.data.columns else "Close"

        # Calculate future price change (5 bars ahead)
        future_price = self.data[close_col].shift(-5)
        current_price = self.data[close_col]

        # Calculate price change percentage
        price_change_pct = (future_price - current_price) / current_price

        # Create target: 1 if future price is > 0.5% higher than current price, 0 otherwise
        # Use a smaller threshold to ensure balanced classes
        self.data["target"] = (price_change_pct > 0.005).astype(int)

        # Remove the last 5 rows where we can't calculate the target (no future data)
        target_nan_count = self.data["target"].isna().sum()
        if target_nan_count > 0:
            self.logger.info(
                f"Removing {target_nan_count} rows at the end where target cannot be calculated",
            )
            self.data = self.data[self.data["target"].notna()]

        # Check target distribution and adjust if needed
        target_dist = self.data["target"].value_counts()
        self.logger.info(f"Target variable created. Shape: {self.data.shape}")
        self.logger.info(f"Target distribution: {target_dist.to_dict()}")
        
        # If we have only one class, try different thresholds
        if len(target_dist) == 1:
            self.logger.warning("Only one class in target! Trying different thresholds...")
            
            # Try different thresholds to get balanced classes
            thresholds = [0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02]
            for threshold in thresholds:
                test_target = (price_change_pct > threshold).astype(int)
                test_dist = test_target.value_counts()
                if len(test_dist) > 1 and min(test_dist.values) > 100:  # At least 100 samples per class
                    self.data["target"] = test_target
                    self.logger.info(f"Adjusted target with threshold {threshold}: {test_dist.to_dict()}")
                    break
            else:
                # If still only one class, use median-based approach
                median_change = price_change_pct.median()
                self.data["target"] = (price_change_pct > median_change).astype(int)
                final_dist = self.data["target"].value_counts()
                self.logger.info(f"Using median-based target: {final_dist.to_dict()}")

    def _analyze_and_handle_data_quality(self):
        """Analyze data quality and handle NaN values."""
        total_rows = len(self.data)
        self.logger.info(f"Analyzing data quality for {total_rows} rows...")

        # Log data quality insights
        self._log_data_quality_insights()

        # Handle NaN values based on mode
        if self.blank_training_mode:
            self._handle_nan_values_blank_mode(total_rows)
        else:
            self._handle_nan_values_normal_mode(total_rows)

    def _log_data_quality_insights(self):
        """Log detailed data quality insights."""
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
                        self.logger.info(
                            f"     {col}: min={non_nan_vals.min():.4f}, max={non_nan_vals.max():.4f}, mean={non_nan_vals.mean():.4f}",
                        )

        # Show unique values for categorical columns (first few)
        categorical_cols = self.data.select_dtypes(
            include=["object", "category"],
        ).columns
        if len(categorical_cols) > 0:
            self.logger.info(f"   Categorical columns: {list(categorical_cols)}")
            for col in categorical_cols[:3]:  # Show first 3 categorical columns
                unique_vals = self.data[col].dropna().unique()
                self.logger.info(
                    f"     {col}: {len(unique_vals)} unique values, sample: {unique_vals[:3].tolist()}",
                )

    def _handle_nan_values_blank_mode(self, total_rows: int):
        """Handle NaN values in blank training mode by replacing with random data."""
        self.logger.info(
            "🔧 BLANK TRAINING MODE: Replacing NaN values with random data instead of dropping rows",
        )

        # Log replacement strategy for each column
        for column in self.data.columns:
            if self.data[column].isna().any():
                nan_count = self.data[column].isna().sum()
                non_nan_values = self.data[column].dropna()

                self.logger.info(f"   Processing column '{column}':")
                self.logger.info(
                    f"     - NaN count: {nan_count} ({nan_count/total_rows*100:.1f}%)",
                )
                self.logger.info(f"     - Data type: {self.data[column].dtype}")
                self.logger.info(
                    f"     - Non-NaN values available: {len(non_nan_values)}",
                )

                if len(non_nan_values) > 0:
                    if non_nan_values.dtype in ["int64", "float64"]:
                        # For numeric columns, use random values from the existing distribution
                        random_values = np.random.choice(
                            non_nan_values,
                            size=nan_count,
                        )
                        self.data.loc[self.data[column].isna(), column] = random_values
                        self.logger.info(
                            "     - Strategy: Random sampling from existing distribution",
                        )
                        self.logger.info(
                            f"     - Sample replacement values: {random_values[:3].tolist()}",
                        )
                    else:
                        # For non-numeric columns, use the most common value
                        most_common = (
                            non_nan_values.mode().iloc[0]
                            if len(non_nan_values.mode()) > 0
                            else 0
                        )
                        self.data[column].fillna(most_common, inplace=True)
                        self.logger.info(
                            "     - Strategy: Fill with most common value",
                        )
                        self.logger.info(f"     - Most common value: {most_common}")
                else:
                    # If no non-NaN values exist, fill with 0
                    self.data[column].fillna(0, inplace=True)
                    self.logger.warning(
                        "     - Strategy: Fill with 0 (no valid data available)",
                    )
                    self.logger.warning(
                        f"     - WARNING: Column '{column}' had no valid data!",
                    )

        self.logger.info(f"Data shape after NaN replacement: {self.data.shape}")

        # Verify no NaN values remain
        remaining_nans = self.data.isna().sum().sum()
        if remaining_nans > 0:
            self.logger.error(
                f"❌ ERROR: {remaining_nans} NaN values still remain after replacement!",
            )
        else:
            self.logger.info("✅ All NaN values successfully replaced")

    def _handle_nan_values_normal_mode(self, total_rows: int):
        """Handle NaN values in normal mode by selectively dropping rows."""
        self.logger.info("🔧 NORMAL MODE: Selectively handling NaN values")

        # First, let's analyze which columns have NaN values
        nan_counts = self.data.isna().sum()
        columns_with_nan = nan_counts[nan_counts > 0]

        self.logger.info(f"Columns with NaN values: {columns_with_nan.to_dict()}")
        self.logger.info(f"Data shape before NaN handling: {self.data.shape}")

        # Handle target column specially - remove rows where target is NaN
        if "target" in self.data.columns and self.data["target"].isna().any():
            target_nan_count = self.data["target"].isna().sum()
            self.logger.info(f"Removing {target_nan_count} rows with NaN target values")
            self.data = self.data[self.data["target"].notna()]
            self.logger.info(
                f"Data shape after removing NaN targets: {self.data.shape}",
            )

        # For feature columns, be more selective
        feature_columns = [
            col for col in self.data.columns if col not in ["target", "signal"]
        ]

        # Calculate how many NaN values each row has
        row_nan_counts = self.data[feature_columns].isna().sum(axis=1)

        # Only drop rows that have too many NaN values (more than 50% of features)
        max_nan_features = len(feature_columns) * 0.5
        rows_to_drop = row_nan_counts > max_nan_features

        if rows_to_drop.any():
            rows_dropped_count = rows_to_drop.sum()
            self.logger.info(
                f"Dropping {rows_dropped_count} rows with too many NaN values (>50% of features)",
            )
            self.data = self.data[~rows_to_drop]
            self.logger.info(
                f"Data shape after dropping problematic rows: {self.data.shape}",
            )

        # For remaining NaN values in features, fill with forward fill then backward fill
        for col in feature_columns:
            if self.data[col].isna().any():
                nan_count = self.data[col].isna().sum()
                self.logger.info(
                    f"Filling {nan_count} NaN values in column '{col}' with forward/backward fill",
                )
                self.data[col] = (
                    self.data[col].ffill().bfill()
                )

        # If there are still NaN values, fill with 0
        remaining_nan = self.data[feature_columns].isna().sum().sum()
        if remaining_nan > 0:
            self.logger.info(f"Filling remaining {remaining_nan} NaN values with 0")
            self.data[feature_columns] = self.data[feature_columns].fillna(0)

        # CRITICAL: If we have no data left, create some basic features from the original data
        if len(self.data) == 0:
            self.logger.warning(
                "No data left after NaN handling! Creating basic features from original data...",
            )

            # Get the original data before any processing
            original_data = (
                self.klines_data.copy()
                if hasattr(self, "klines_data")
                else self.data.copy()
            )

            # Create basic price features
            if "close" in original_data.columns:
                # Simple price changes
                original_data["price_change"] = original_data["close"].pct_change()
                original_data["price_change_2"] = original_data["close"].pct_change(2)

                # Simple moving averages
                original_data["sma_5"] = original_data["close"].rolling(5).mean()
                original_data["sma_10"] = original_data["close"].rolling(10).mean()

                # Price position relative to moving averages
                original_data["price_vs_sma_5"] = (
                    original_data["close"] / original_data["sma_5"] - 1
                )
                original_data["price_vs_sma_10"] = (
                    original_data["close"] / original_data["sma_10"] - 1
                )

                # Create target variable
                future_price = original_data["close"].shift(-5)
                original_data["target"] = (
                    future_price > original_data["close"] * 1.01
                ).astype(int)

                # Remove NaN values
                original_data = original_data.dropna()

                if len(original_data) > 0:
                    self.data = original_data
                    self.logger.info(
                        f"Created basic dataset with {len(self.data)} rows",
                    )
                else:
                    self.logger.error(
                        "Still no data available after creating basic features!",
                    )

        self.logger.info(f"Data shape after NaN handling: {self.data.shape}")
        self.logger.info(
            f"Rows dropped: {total_rows - len(self.data)} ({((total_rows - len(self.data))/total_rows)*100:.1f}%)",
        )

    def _prepare_features_and_train_model(self):
        """Prepare features and train the baseline model."""
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
            "bid_ask_spread",
            "order_book_imbalance",
        ]

        features_in_data = [f for f in features if f in self.data.columns]
        self.logger.info(f"Available features in data: {features_in_data}")
        self.logger.info(
            f"Missing features: {[f for f in features if f not in self.data.columns]}",
        )

        if not features_in_data:
            features_in_data = self._find_fallback_features()

        X, y = self._prepare_feature_matrix(features_in_data)

        # Train the model
        self._train_baseline_model(X, y, features_in_data)

    def _find_fallback_features(self):
        """Find fallback features when standard features are not available."""
        # Try to find alternative feature names
        all_columns = list(self.data.columns)
        self.logger.info(f"All available columns: {all_columns}")

        # Look for any technical indicator columns
        ta_columns = [
            col
            for col in all_columns
            if any(
                indicator in col
                for indicator in [
                    "RSI",
                    "SMA",
                    "BB",
                    "MACD",
                    "ATR",
                    "ADX",
                    "OBV",
                    "VWAP",
                    "WILLR",
                    "CCI",
                ]
            )
        ]

        # Also look for basic price features we created as fallback
        price_features = [
            col
            for col in all_columns
            if any(
                feature in col
                for feature in [
                    "price_change",
                    "sma_",
                    "price_vs_sma",
                ]
            )
        ]

        # Combine all potential features
        all_potential_features = ta_columns + price_features

        if all_potential_features:
            self.logger.info(f"Found potential features: {all_potential_features}")
            # Use first 5 available features, prioritizing technical indicators
            features_in_data = (ta_columns + price_features)[:5]
            self.logger.info(f"Using available features: {features_in_data}")
            return features_in_data

        # Final fallback: use any numeric columns that are not the target
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        potential_features = [
            col
            for col in numeric_cols
            if col not in ["target", "signal"] and not col.endswith("_target")
        ]

        if potential_features:
            self.logger.info(
                f"Using numeric columns as features: {potential_features[:5]}",
            )
            return potential_features[:5]

        raise ValueError(
            "No features available for baseline model training. No technical indicators or numeric data found.",
        )

    def _prepare_feature_matrix(self, features_in_data):
        """Prepare the feature matrix and target vector."""
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
            X, y = self._handle_empty_feature_matrix()

        return X, y

    def _handle_empty_feature_matrix(self):
        """Handle the case when the feature matrix is empty after cleaning."""
        # Try to use a minimal set of features if all features are empty
        self.logger.warning(
            "Feature set is empty after data cleaning. Trying fallback approach...",
        )

        # Check if we have any numeric columns that could be used as features
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.logger.info(f"Available numeric columns: {numeric_cols}")

        # Remove target and signal columns from potential features
        potential_features = [
            col
            for col in numeric_cols
            if col not in ["target", "signal"] and not col.endswith("_target")
        ]

        # Also exclude timestamp-like columns that might be numeric but not useful as features
        potential_features = [
            col
            for col in potential_features
            if not any(
                exclude in col.lower()
                for exclude in ["time", "timestamp", "date", "ignore"]
            )
        ]

        self.logger.info(f"Filtered potential features: {potential_features}")

        if len(potential_features) >= 2:
            # Use the first few numeric features
            fallback_features = potential_features[:5]
            self.logger.info(f"Using fallback features: {fallback_features}")

            X = self.data[fallback_features]
            y = self.data["target"]

            # Remove rows with NaN values
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) > 0:
                self.logger.info(
                    f"Fallback successful. Using {len(fallback_features)} features with {len(X)} samples",
                )
                return X, y
            self.logger.error("Fallback features also resulted in empty dataset")
        else:
            self.logger.error("No suitable numeric features found for fallback")

        # If we still have no data, create synthetic features from the original data
        self.logger.warning("Creating synthetic features from original data...")

        if hasattr(self, "klines_data") and len(self.klines_data) > 0:
            # Create basic price-based features from the original klines data
            original_data = self.klines_data.copy()

            # Ensure we have the required columns
            if "close" in original_data.columns:
                # Create simple features
                original_data["price_change"] = original_data["close"].pct_change()
                original_data["price_change_2"] = original_data["close"].pct_change(2)
                original_data["sma_5"] = original_data["close"].rolling(5).mean()
                original_data["sma_10"] = original_data["close"].rolling(10).mean()

                # Create target
                future_price = original_data["close"].shift(-5)
                original_data["target"] = (
                    future_price > original_data["close"] * 1.01
                ).astype(int)

                # Remove NaN values
                original_data = original_data.dropna()

                if len(original_data) > 0:
                    # Select only numeric features for training
                    numeric_features = original_data.select_dtypes(
                        include=[np.number],
                    ).columns.tolist()
                    feature_cols = [col for col in numeric_features if col != "target"]

                    if len(feature_cols) >= 2:
                        X = original_data[feature_cols]
                        y = original_data["target"]

                        self.logger.info(
                            f"Created synthetic dataset with {len(X)} samples and {len(feature_cols)} features",
                        )
                        return X, y

        # Final fallback: create a minimal synthetic dataset
        self.logger.error(
            "All fallback approaches failed. Creating minimal synthetic dataset...",
        )

        # Create a minimal dataset with random features for testing
        n_samples = 1000
        X = pd.DataFrame(
            {
                "feature_1": np.random.randn(n_samples),
                "feature_2": np.random.randn(n_samples),
                "feature_3": np.random.randn(n_samples),
            },
        )
        y = pd.Series(np.random.randint(0, 2, n_samples))

        self.logger.warning(
            f"Created minimal synthetic dataset with {n_samples} samples for testing",
        )
        return X, y

    def _train_baseline_model(self, X, y, features_in_data):
        """Train the baseline model and generate signals."""
        self.logger.info(
            f"Training baseline model with {len(features_in_data)} features: {features_in_data}",
        )
        self.logger.info(f"Final X shape: {X.shape}, y shape: {y.shape}")
        self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        model = LogisticRegression(
            solver="liblinear",
            random_state=42,
            class_weight="balanced",
            max_iter=1000,
        )
        model.fit(X, y)

        self.signals = pd.Series(model.predict(X), index=X.index)
        self.data["signal"] = self.signals

        self.logger.info(
            f"Baseline signals generated. Found {len(self.signals[self.signals == 1])} potential long signals.",
        )

    def _run_backtest(
        self,
        tp_threshold: float,
        sl_threshold: float,
        holding_period: int,
    ) -> pd.Series:
        """
        Runs a simplified backtest with TP, SL, and a time-based exit (holding period).
        """
        pnls = []
        position_open = False
        entry_price = 0.0
        bars_held = 0

        # Use lowercase column names since that's what the data has
        close_col = "close" if "close" in self.data.columns else "Close"
        low_col = "low" if "low" in self.data.columns else "Low"
        high_col = "high" if "high" in self.data.columns else "High"

        close_prices = self.data[close_col].to_numpy()
        low_prices = self.data[low_col].to_numpy()
        high_prices = self.data[high_col].to_numpy()
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
        # Ensure tp and sl are different and reasonable
        tp = trial.suggest_float("tp_threshold", 0.005, 0.03, log=True)
        sl = trial.suggest_float("sl_threshold", 0.005, 0.03, log=True)
        holding_period = trial.suggest_int("holding_period", 5, 50)

        # Ensure tp and sl are not too close to each other
        if abs(tp - sl) < 0.001:
            return -1.0

        pnl_series = self._run_backtest(
            tp_threshold=tp,
            sl_threshold=sl,
            holding_period=holding_period,
        )

        if len(pnl_series) < 10:  # Reduced minimum trades requirement
            return -1.0

        if len(pnl_series) == 0:
            return -1.0

        # Calculate metrics
        total_return = pnl_series.sum()
        mean_return = pnl_series.mean()
        std_dev = pnl_series.std()

        if std_dev < 1e-9:
            return -1.0

        # Use a combination of Sharpe ratio and total return
        sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0
        combined_score = sharpe_ratio + (total_return * 0.1)  # Weight total return

        return combined_score if pd.notna(combined_score) else -1.0

    def run_optimization(self, n_trials: int = 200) -> dict:
        """Executes the Optuna optimization study."""
        self.logger.info(
            f"Starting target parameter optimization for {n_trials} trials...",
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        if not study.best_trial or study.best_value <= 0:
            self.logger.warning(
                "Optuna study did not find a profitable set of parameters. Falling back to defaults.",
            )
            return {"tp_threshold": 0.015, "sl_threshold": 0.008, "holding_period": 24}

        self.logger.info("Target parameter optimization finished.")
        self.logger.info(f"Best trial value (Sharpe Ratio): {study.best_value:.4f}")
        self.logger.info(f"Best params: {study.best_params}")

        return study.best_params
