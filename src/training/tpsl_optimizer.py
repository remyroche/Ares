# src/training/tpsl_optimizer.py

import numba
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.analyst.predictive_ensembles.ensemble_orchestrator import (
    RegimePredictiveEnsembles,
)
from src.database.sqlite_manager import SQLiteManager
from src.utils.logger import logger

# Suppress Optuna's informational messages for a cleaner log
optuna.logging.set_verbosity(optuna.logging.WARNING)


@numba.jit(nopython=True, cache=True)
def _numba_backtest(
    close_prices: np.ndarray,
    low_prices: np.ndarray,
    high_prices: np.ndarray,
    signals: np.ndarray,
    ml_sell_confidence: np.ndarray,
    tp_threshold: float,
    sl_threshold: float,
    enable_ml_early_exit: bool,
    early_exit_confidence: float,
) -> np.ndarray:
    """
    A Numba-accelerated backtesting loop.

    Args:
        close_prices (np.ndarray): Array of close prices.
        low_prices (np.ndarray): Array of low prices.
        high_prices (np.ndarray): Array of high prices.
        signals (np.ndarray): Array of trading signals (1 for long, 0 for hold).
        ml_sell_confidence (np.ndarray): Confidence scores for ML-based sell predictions.
        tp_threshold (float): Take profit percentage.
        sl_threshold (float): Stop loss percentage.
        enable_ml_early_exit (bool): Flag to enable/disable ML early exits.
        early_exit_confidence (float): Confidence threshold for ML early exits.

    Returns:
        np.ndarray: An array of PnL values for each closed trade.
    """
    pnls = []
    position_open = False
    entry_price = 0.0

    for i in range(1, len(close_prices)):
        # Entry Condition: Open a new long position if not already in one
        if not position_open and signals[i - 1] == 1:
            position_open = True
            entry_price = close_prices[i - 1]

        # Exit Conditions: If a position is open
        if position_open:
            # 1. ML-based Early Exit (adverse movement)
            if (
                enable_ml_early_exit
                and ml_sell_confidence[i] > early_exit_confidence
            ):
                pnl = (close_prices[i] - entry_price) / entry_price
                pnls.append(pnl)
                position_open = False
                continue

            # 2. Take Profit Exit
            if high_prices[i] >= entry_price * (1 + tp_threshold):
                pnls.append(tp_threshold)
                position_open = False
                continue

            # 3. Stop Loss Exit
            if low_prices[i] <= entry_price * (1 - sl_threshold):
                pnls.append(-sl_threshold)
                position_open = False

    return np.array(pnls, dtype=np.float64)


class TpSlOptimizer:
    """
    Optimizes Take Profit (TP) and Stop Loss (SL) thresholds using Optuna.

    This version is computationally efficient, using Numba for backtesting, and
    focuses the optimization objective on profitability (Profit Factor) while
    using risk metrics as constraints.
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
        self.data = pd.DataFrame()
        self._prepare_data_and_signals()

    def _prepare_data_and_signals(self):
        """
        Loads data, engineers features, and generates baseline trading signals.
        """
        logger.info("Preparing data and generating signals for optimization...")

        # 1. Load and prepare data
        table_name = f"{self.symbol}_{self.timeframe}"
        self.data = self.db_manager.get_all_data(table_name)
        if self.data.empty:
            raise ValueError(f"No data for {table_name}.")

        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self.data.set_index("timestamp", inplace=True)
        self.data.rename(
            columns={c: c.capitalize() for c in ["open", "high", "low", "close", "volume"]},
            inplace=True,
        )

        # 2. Feature Engineering
        self.data.ta.rsi(length=14, append=True)
        self.data.ta.macd(append=True)
        self.data.ta.bbands(length=20, append=True)
        self.data.ta.atr(length=14, append=True)
        self.data.ta.adx(length=14, append=True)

        # 3. Create a simple binary target for the baseline model
        self.data["target"] = (self.data["Close"].shift(-5) > self.data["Close"] * 1.01).astype(int)
        self.data.dropna(inplace=True)

        features = [
            "RSI_14", "MACD_12_26_9", "BBU_20_2.0",
            "ATRr_14", "ADX_14"
        ]
        features_in_data = [f for f in features if f in self.data.columns]
        if not features_in_data:
            raise ValueError("No features available for model training.")

        X = self.data[features_in_data]
        y = self.data["target"]

        # 4. Train a simple model (Logistic Regression) to generate signals
        model = LogisticRegression(solver="liblinear", random_state=42, class_weight="balanced")
        model.fit(X, y)
        self.data["signal"] = model.predict(X)
        
        # 5. Prepare ML confidence data for early exit signals
        self._prepare_ml_exit_data()

        logger.info(f"Data prepared. Found {self.data['signal'].sum()} potential long signals.")

    def _prepare_ml_exit_data(self):
        """
        Generates ML-based confidence scores for early exits.
        Uses a vectorized momentum calculation as a fallback.
        """
        logger.info("Generating ML confidence scores for early exit analysis...")
        # Fallback: simple momentum-based predictions (vectorized)
        momentum = self.data["Close"].pct_change(5).fillna(0)
        
        # Conditions for sell signal (strong negative momentum)
        is_sell_signal = momentum < -0.01
        
        # Populate confidence score. Only strong negative momentum gets a high score.
        self.data["ml_sell_confidence"] = np.where(is_sell_signal, 0.75, 0.0)

    def _run_backtest(
        self,
        tp_threshold: float,
        sl_threshold: float,
        enable_ml_early_exit: bool,
        early_exit_confidence: float,
    ) -> pd.Series:
        """
        Wrapper for the Numba-accelerated backtest function.
        """
        # Extract required data as NumPy arrays for Numba
        close_prices = self.data["Close"].to_numpy()
        low_prices = self.data["Low"].to_numpy()
        high_prices = self.data["High"].to_numpy()
        signals = self.data["signal"].to_numpy()
        ml_sell_confidence = self.data["ml_sell_confidence"].to_numpy()
        
        # Run the highly optimized backtest
        pnls = _numba_backtest(
            close_prices,
            low_prices,
            high_prices,
            signals,
            ml_sell_confidence,
            tp_threshold,
            sl_threshold,
            enable_ml_early_exit,
            early_exit_confidence,
        )
        return pd.Series(pnls)

    def _calculate_max_drawdown(self, pnl_series: pd.Series) -> float:
        """Calculate maximum drawdown from PnL series."""
        if pnl_series.empty:
            return 0.0
        cumulative = (1 + pnl_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        The objective function for Optuna to maximize.

        Focuses on maximizing Profit Factor, a key measure of profitability,
        while penalizing for excessive drawdown.
        """
        # Suggest parameters for the trial
        tp = trial.suggest_float("tp_threshold", 0.005, 0.1, log=True)
        sl = trial.suggest_float("sl_threshold", 0.005, 0.1, log=True)
        early_exit_confidence = trial.suggest_float("early_exit_confidence", 0.5, 0.95)
        enable_ml_early_exit = trial.suggest_categorical("enable_ml_early_exit", [True, False])

        # Run the backtest with the trial parameters
        pnl_series = self._run_backtest(
            tp_threshold=tp,
            sl_threshold=sl,
            enable_ml_early_exit=enable_ml_early_exit,
            early_exit_confidence=early_exit_confidence,
        )

        # Prune trial if not enough trades for statistical significance
        if len(pnl_series) < 20:
            return -1.0

        # Calculate performance metrics
        gross_profit = pnl_series[pnl_series > 0].sum()
        gross_loss = abs(pnl_series[pnl_series < 0].sum())

        if gross_loss < 1e-9: # Avoid division by zero
            return gross_profit * 10 if gross_profit > 0 else 0

        profit_factor = gross_profit / gross_loss
        
        # Risk Constraint: Penalize if drawdown exceeds a reasonable threshold
        max_drawdown = self._calculate_max_drawdown(pnl_series)
        ACCEPTABLE_DRAWDOWN = 0.25
        if max_drawdown > ACCEPTABLE_DRAWDOWN:
            # Apply a penalty proportional to the excess drawdown
            return profit_factor - (max_drawdown * 10) 
        
        # Final Score: Favor strategies with a good profit factor that also trade frequently
        final_score = profit_factor * np.log1p(len(pnl_series))
        
        return final_score if pd.notna(final_score) else -1.0

    def run_optimization(self, n_trials: int = 200) -> dict:
        """
        Executes the Optuna optimization study.
        """
        logger.info(f"Starting TP/SL optimization for {n_trials} trials...")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        if not study.best_trial or study.best_value < 0:
            logger.warning("Optuna could not find a profitable parameter set. Returning defaults.")
            return {
                "tp_threshold": 0.02,
                "sl_threshold": 0.01,
                "early_exit_confidence": 0.75,
                "enable_ml_early_exit": True,
            }

        logger.info("TP/SL optimization finished.")
        logger.info(f"Best trial score (Profit Factor based): {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params
