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
    ml_buy_confidence: np.ndarray,
    ml_sell_confidence: np.ndarray,
    tp_long: float,
    sl_long: float,
    tp_short: float,
    sl_short: float,
    enable_ml_early_exit: bool,
    early_exit_confidence: float,
) -> np.ndarray:
    """
    A Numba-accelerated backtesting loop for both long and short trades,
    including asymmetrical barriers and trading fees. Returns an array of
    [pnl, direction] for each trade.
    """
    trades = []
    position_direction = 0  # 0: No position, 1: Long, -1: Short
    entry_price = 0.0
    TRADE_FEE = 0.0008  # Assumed 0.08% fee per trade (buy + sell)

    for i in range(1, len(close_prices)):
        # Entry Conditions
        if position_direction == 0:
            if signals[i - 1] == 1:  # Go Long
                position_direction = 1
                entry_price = close_prices[i - 1]
            elif signals[i - 1] == -1:  # Go Short
                position_direction = -1
                entry_price = close_prices[i - 1]

        # Exit Conditions for an OPEN LONG position
        if position_direction == 1:
            # ML Early Exit
            if enable_ml_early_exit and ml_sell_confidence[i] > early_exit_confidence:
                pnl = (close_prices[i] - entry_price) / entry_price - TRADE_FEE
                trades.append((pnl, 1))
                position_direction = 0
                continue
            # Take Profit
            if high_prices[i] >= entry_price * (1 + tp_long):
                trades.append((tp_long - TRADE_FEE, 1))
                position_direction = 0
                continue
            # Stop Loss
            if low_prices[i] <= entry_price * (1 - sl_long):
                trades.append((-sl_long - TRADE_FEE, 1))
                position_direction = 0

        # Exit Conditions for an OPEN SHORT position
        elif position_direction == -1:
            # ML Early Exit
            if enable_ml_early_exit and ml_buy_confidence[i] > early_exit_confidence:
                pnl = (entry_price - close_prices[i]) / entry_price - TRADE_FEE
                trades.append((pnl, -1))
                position_direction = 0
                continue
            # Take Profit
            if low_prices[i] <= entry_price * (1 - tp_short):
                trades.append((tp_short - TRADE_FEE, -1))
                position_direction = 0
                continue
            # Stop Loss
            if high_prices[i] >= entry_price * (1 + sl_short):
                trades.append((-sl_short - TRADE_FEE, -1))
                position_direction = 0

    if not trades:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(trades, dtype=np.float64)


class TpSlOptimizer:
    """
    Optimizes asymmetrical Take Profit (TP) and Stop Loss (SL) thresholds
    for LONG & SHORT strategies, including trading fees.
    """

    def __init__(self, db_manager: SQLiteManager, symbol: str, timeframe: str):
        self.db_manager = db_manager
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = pd.DataFrame()
        self._prepare_data_and_signals()

    def _prepare_data_and_signals(self):
        logger.info("Preparing data and generating signals for optimization...")

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

        # Feature Engineering
        self.data.ta.rsi(length=14, append=True)
        self.data.ta.macd(append=True)
        self.data.ta.bbands(length=20, append=True)
        self.data.ta.atr(length=14, append=True)
        self.data.ta.adx(length=14, append=True)

        # Create a three-class target for Long (1), Short (-1), and Hold (0)
        future_price = self.data["Close"].shift(-5)
        price_change = (future_price - self.data["Close"]) / self.data["Close"]
        
        PROFIT_THRESHOLD = 0.01 
        
        conditions = [
            price_change > PROFIT_THRESHOLD,
            price_change < -PROFIT_THRESHOLD,
        ]
        choices = [1, -1]
        self.data["target"] = np.select(conditions, choices, default=0)

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

        model = LogisticRegression(solver="liblinear", random_state=42, class_weight="balanced")
        model.fit(X, y)
        self.data["signal"] = model.predict(X)
        
        self._prepare_ml_exit_data()

        logger.info(
            f"Data prepared. Found {len(self.data[self.data['signal'] == 1])} long signals "
            f"and {len(self.data[self.data['signal'] == -1])} short signals."
        )

    def _prepare_ml_exit_data(self):
        logger.info("Generating ML confidence scores for early exit analysis...")
        momentum = self.data["Close"].pct_change(5).fillna(0)
        
        self.data["ml_sell_confidence"] = np.where(momentum < -0.01, 0.75, 0.0)
        self.data["ml_buy_confidence"] = np.where(momentum > 0.01, 0.75, 0.0)

    def _run_backtest(
        self,
        tp_long: float,
        sl_long: float,
        tp_short: float,
        sl_short: float,
        enable_ml_early_exit: bool,
        early_exit_confidence: float,
    ) -> pd.DataFrame:
        pnl_array = _numba_backtest(
            self.data["Close"].to_numpy(),
            self.data["Low"].to_numpy(),
            self.data["High"].to_numpy(),
            self.data["signal"].to_numpy(),
            self.data["ml_buy_confidence"].to_numpy(),
            self.data["ml_sell_confidence"].to_numpy(),
            tp_long,
            sl_long,
            tp_short,
            sl_short,
            enable_ml_early_exit,
            early_exit_confidence,
        )
        if pnl_array.shape[0] == 0:
            return pd.DataFrame(columns=['pnl', 'direction'])
        return pd.DataFrame(pnl_array, columns=['pnl', 'direction'])

    def _calculate_max_drawdown(self, pnl_series: pd.Series) -> float:
        if pnl_series.empty:
            return 0.0
        cumulative = (1 + pnl_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def _calculate_performance_metrics(self, pnl_series: pd.Series) -> dict:
        if pnl_series.empty:
            return {
                "gross_profit": 0, "gross_loss": 0, "profit_factor": 0,
                "max_drawdown": 0, "trade_count": 0,
            }

        gross_profit = pnl_series[pnl_series > 0].sum()
        gross_loss = abs(pnl_series[pnl_series < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")
        max_drawdown = self._calculate_max_drawdown(pnl_series)
        
        return {
            "gross_profit": gross_profit, "gross_loss": gross_loss,
            "profit_factor": profit_factor, "max_drawdown": max_drawdown,
            "trade_count": len(pnl_series),
        }

    def objective(self, trial: optuna.trial.Trial) -> float:
        tp_long = trial.suggest_float("tp_long", 0.005, 0.1, log=True)
        sl_long = trial.suggest_float("sl_long", 0.005, 0.1, log=True)
        tp_short = trial.suggest_float("tp_short", 0.005, 0.1, log=True)
        sl_short = trial.suggest_float("sl_short", 0.005, 0.1, log=True)
        
        early_exit_confidence = trial.suggest_float("early_exit_confidence", 0.5, 0.95)
        enable_ml_early_exit = trial.suggest_categorical("enable_ml_early_exit", [True, False])

        results_df = self._run_backtest(
            tp_long=tp_long, sl_long=sl_long, tp_short=tp_short, sl_short=sl_short,
            enable_ml_early_exit=enable_ml_early_exit,
            early_exit_confidence=early_exit_confidence,
        )

        if len(results_df) < 25:
            return -1.0

        # Calculate metrics for longs, shorts, and total
        pnl_total = results_df['pnl']
        pnl_longs = results_df[results_df['direction'] == 1]['pnl']
        pnl_shorts = results_df[results_df['direction'] == -1]['pnl']

        metrics_total = self._calculate_performance_metrics(pnl_total)
        metrics_longs = self._calculate_performance_metrics(pnl_longs)
        metrics_shorts = self._calculate_performance_metrics(pnl_shorts)

        # Store detailed metrics for later analysis
        trial.set_user_attr("total", metrics_total)
        trial.set_user_attr("long", metrics_longs)
        trial.set_user_attr("short", metrics_shorts)

        # Main objective score calculation
        profit_factor = metrics_total["profit_factor"]
        max_drawdown = metrics_total["max_drawdown"]
        
        ACCEPTABLE_DRAWDOWN = 0.25
        if max_drawdown > ACCEPTABLE_DRAWDOWN:
            return profit_factor - (max_drawdown * 10) 
        
        final_score = profit_factor * np.log1p(metrics_total["trade_count"])
        
        return final_score if pd.notna(final_score) else -1.0

    def run_optimization(self, n_trials: int = 250) -> dict:
        logger.info(f"Starting asymmetrical TP/SL optimization for {n_trials} trials...")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        logger.info("TP/SL optimization finished.")

        best_trial = study.best_trial
        if not best_trial or best_trial.value < 0:
            logger.warning("Optuna could not find a profitable parameter set. Returning defaults.")
            return {
                "tp_long": 0.02, "sl_long": 0.01, "tp_short": 0.02, "sl_short": 0.01,
                "early_exit_confidence": 0.75, "enable_ml_early_exit": True,
            }

        logger.info(f"Best trial score (Profit Factor based): {best_trial.value:.4f}")
        logger.info(f"Best params: {best_trial.params}")

        # Log detailed performance metrics for the best trial
        logger.info("--- Best Trial Performance Breakdown ---")
        for category in ["total", "long", "short"]:
            attrs = best_trial.user_attrs.get(category, {})
            # Check if there were any trades in this category before logging
            if attrs.get("trade_count", 0) > 0:
                logger.info(f"  {category.upper()} Trades ({attrs['trade_count']}):")
                logger.info(f"    - Profit Factor: {attrs.get('profit_factor', 0):.2f}")
                logger.info(f"    - Gross Profit: {attrs.get('gross_profit', 0):.4f}")
                logger.info(f"    - Gross Loss: {attrs.get('gross_loss', 0):.4f}")
                logger.info(f"    - Max Drawdown: {attrs.get('max_drawdown', 0):.2%}")
        
        return best_trial.params
