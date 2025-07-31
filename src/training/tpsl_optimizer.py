# src/training/tpsl_optimizer.py

import optuna
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression

from src.database.sqlite_manager import SQLiteManager
from src.utils.logger import logger

# Import the existing TechnicalAnalyzer to use its methods directly
from src.analyst.technical_analyzer import TechnicalAnalyzer

# Suppress Optuna's informational messages for a cleaner log during optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TpSlOptimizer:
    """
    Optimizes Take Profit (TP) and Stop Loss (SL) thresholds using Optuna.

    This class uses a simplified backtesting heuristic. It first trains a simple
    ML model to generate trading signals, and then uses Optuna to find the
    optimal TP/SL levels that maximize a performance metric (e.g., Sharpe Ratio)
    on those signals.
    """

    def __init__(self, db_manager: SQLiteManager, symbol: str, timeframe: str):
        """
        Initializes the optimizer.

        Args:
            db_manager (SQLiteManager): The database manager to fetch data.
            symbol (str): The trading symbol (e.g., 'BTCUSDT').
            timeframe (str): The timeframe for the data (e.g., '1h').
        """
        self.db_manager = db_manager
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = None
        self.signals = None
        self._prepare_data_and_signals()

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds standard technical indicators to the dataframe."""
        logger.info("Adding standard technical indicators...")
        data.ta.rsi(length=14, append=True)
        data.ta.sma(length=50, append=True)
        data.ta.bbands(length=20, append=True)
        data.ta.macd(append=True)
        data.ta.atr(length=14, append=True)
        data.ta.adx(length=14, append=True)
        data.ta.obv(append=True)
        data.ta.vwap(append=True)
        return data

    def _prepare_data_and_signals(self):
        """
        Loads data, engineers features, and trains a simple model to get signals.
        This provides a baseline of trading intentions to test TP/SL levels against.
        """
        logger.info(
            "Preparing data and generating baseline signals for TP/SL optimization..."
        )

        # 1. Load data from the database
        table_name = f"{self.symbol}_{self.timeframe}"
        self.data = self.db_manager.get_all_data(table_name)
        if self.data.empty:
            raise ValueError(f"No data found for {table_name} in the database.")

        # Ensure timestamp is a datetime object and set as index
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self.data.set_index("timestamp", inplace=True)

        # Rename columns to be compatible with pandas_ta
        self.data.rename(
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

        # 2. Comprehensive Feature Engineering for the baseline model
        logger.info("Adding a comprehensive set of features for the baseline model...")

        # Add standard TAs
        self.data = self._add_technical_indicators(self.data)

        # Use TechnicalAnalyzer for Order Book features
        logger.info("Using TechnicalAnalyzer for order book features...")
        try:
            technical_analyzer = TechnicalAnalyzer(self.data)
            # Assuming a method name like 'add_order_book_features' exists
            self.data = technical_analyzer.add_order_book_features()
        except Exception as e:
            logger.error(
                f"Could not add order book features using TechnicalAnalyzer: {e}"
            )

        # 3. Create a simple binary target for the baseline model
        self.data["target"] = (
            self.data["Close"].shift(-5) > self.data["Close"] * 1.01
        ).astype(int)

        # Drop NaNs created by feature engineering and target creation
        self.data.dropna(inplace=True)

        # Use a comprehensive feature set for the baseline model
        # Column names are based on pandas_ta defaults
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
            "order_book_imbalance",  # From TechnicalAnalyzer
        ]

        features_in_data = [f for f in features if f in self.data.columns]
        if not features_in_data:
            raise ValueError("No features available for model training.")
        logger.info(
            f"Using the following features for baseline model: {features_in_data}"
        )

        target = "target"
        X = self.data[features_in_data]
        y = self.data[target]

        if X.empty:
            raise ValueError("Feature set is empty. Cannot train baseline model.")

        # 4. Train a simple model (Logistic Regression) to generate signals
        model = LogisticRegression(
            solver="liblinear", random_state=42, class_weight="balanced", max_iter=1000
        )
        model.fit(X, y)

        # 5. Generate signals (1 for long, 0 for hold)
        predictions = model.predict(X)
        self.signals = pd.Series(predictions, index=X.index)
        self.data["signal"] = self.signals

        logger.info(
            f"Baseline signals generated. Found {len(self.signals[self.signals == 1])} potential long signals."
        )

    def _run_backtest(self, tp_threshold: float, sl_threshold: float) -> pd.Series:
        """
        Runs a simplified iterative backtest based on pre-generated signals.
        """
        pnls = []
        position_open = False
        entry_price = 0.0

        close_prices = self.data["Close"].to_numpy()
        low_prices = self.data["Low"].to_numpy()
        high_prices = self.data["High"].to_numpy()
        signals = self.data["signal"].to_numpy()

        for i in range(1, len(self.data)):
            if not position_open and signals[i - 1] == 1:
                position_open = True
                entry_price = close_prices[i - 1]

            if position_open:
                if high_prices[i] >= entry_price * (1 + tp_threshold):
                    pnls.append(tp_threshold)
                    position_open = False
                elif low_prices[i] <= entry_price * (1 - sl_threshold):
                    pnls.append(-sl_threshold)
                    position_open = False

        return pd.Series(pnls)

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        The objective function for Optuna to maximize.
        """
        tp = trial.suggest_float("tp_threshold", 0.002, 0.05, log=True)
        sl = trial.suggest_float("sl_threshold", 0.002, 0.05, log=True)

        pnl_series = self._run_backtest(tp_threshold=tp, sl_threshold=sl)

        if len(pnl_series) < 20:
            return -1.0

        std_dev = pnl_series.std()
        if std_dev < 1e-9:
            return 0.0

        sharpe_ratio = pnl_series.mean() / std_dev
        return sharpe_ratio if pd.notna(sharpe_ratio) else -1.0

    def run_optimization(self, n_trials: int = 150) -> dict:
        """
        Executes the Optuna optimization study.
        """
        logger.info(f"Starting TP/SL optimization for {n_trials} trials...")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        if not study.best_trial or study.best_value < 0:
            logger.warning(
                "Optuna study did not find a profitable set of TP/SL parameters."
            )
            return {"tp_threshold": 0.01, "sl_threshold": 0.01}

        logger.info("TP/SL optimization finished.")
        logger.info(f"Best trial value (Sharpe Ratio): {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params
