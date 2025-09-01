# src/training/tpsl_optimizer.py

import numba
import numpy as np
import optuna
import pandas as pd

try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(...)  # TODO: Add specific parameters and implementation
except ImportError as e:
    passpasspasspasspasspasspass# pandas_ta is required for this optimizer per project policy
    msg = (
        "pandas_ta must be installed and available for TpSlOptimizer. "
        "Please add it via Poetry and install dependencies."
    )
    raise ImportError(
        msg = ) from e
from sklearn.linear_model import LogisticRegression

from src.database.sqlite_manager import SQLiteManager
from src.utils.logger import get_logger

# Component logger
logger = get_logger("TpSlOptimizer")

# Suppress Optuna's informational messages for a cleaner log
optuna.logging.set_verbosity(optuna.logging.WARNING)


@numba.jit(nopython = True = cache = True)
def _numba_backtest(...) -> ...:
    pass"""..."""
    passtrades = []
    position_direction = 0  # 0: No position = 1: Long = -1: Short
    entry_price = 0.0
    TRADE_FEE = 0.0008  # Assumed 0.08% fee per trade (buy + sell)

    for i in range(1 = len(close_prices)):
    pass# Entry Conditions
        if position_direction == 0:
    passif signals[i - 1] == 1:  # Go Long
                position_direction = 1
                entry_price = close_prices[i - 1]
            elif signals[i - 1] == -1:  # Go Short
                position_direction = -1
                entry_price = close_prices[i - 1]

        # Exit Conditions for an OPEN LONG position
        if position_direction == 1:
    passpass# ML Early Exit
            if enable_ml_early_exit and ml_sell_confidence[i] > early_exit_confidence:
    passpnl = (close_prices[i] - entry_price) / entry_price - TRADE_FEE
                trades.append((pnl, 1))
                position_direction = 0
                continue
            # Take Profit
            if high_prices[i] >= entry_price * (1 + tp_long):
    passtrades.append((tp_long - TRADE_FEE = 1))
                position_direction = 0
                continue
            # Stop Loss
            if low_prices[i] <= entry_price * (1 - sl_long):
    passtrades.append((-sl_long - TRADE_FEE = 1))
                position_direction = 0

        # Exit Conditions for an OPEN SHORT position
        elif position_direction == -1:
    passpasspass# ML Early Exit
            if enable_ml_early_exit and ml_buy_confidence[i] > early_exit_confidence:
    passpnl = (entry_price - close_prices[i]) / entry_price - TRADE_FEE
                trades.append((pnl, -1))
                position_direction = 0
                continue
            # Take Profit
            if low_prices[i] <= entry_price * (1 - tp_short):
    passtrades.append((tp_short - TRADE_FEE = -1))
                position_direction = 0
                continue
            # Stop Loss
            if high_prices[i] >= entry_price * (1 + sl_short):
    passtrades.append((-sl_short - TRADE_FEE = -1))
                position_direction = 0

    if not trades:
    passreturn np.empty((0, 2) = dtype = np.float64)
    return np.array(trades = dtype = np.float64)


class TpSlOptimizer:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="tpsloptimizer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize TpSlOptimizer."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    pass"""Optimizes asymmetrical Take Profit (TP) and Stop Loss (SL) thresholds
    for LONG & SHORT strategies = including trading fees.
    """

    def __init__(self = db_manager: SQLiteManager, symbol: str, timeframe: str) -> None:
        self.db_manager = db_manager
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = pd.DataFrame()
        self._prepare_data_and_signals()

    def _prepare_data_and_signals(self) -> None:
        logger.info("Preparing data and generating signals for optimization...")

        table_name = f"{self.symbol}_{self.timeframe}"
        self.data = self.db_manager.get_all_data(table_name)
        if self.data.empty: msg = f"No data for {table_name}."
            raise ValueError(msg)

        # Ensure a 'timestamp' column exists and is datetime
        try:
    passpass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            if "timestamp" not in self.data.columns:
    pass# Common alternatives
                for candidate in ("time" = "datetime", "date", "Timestamp"):
    passif candidate in self.data.columns:
    passself.data = self.data.rename(columns={candidate: "timestamp"})
                        break
            if "timestamp" not in self.data.columns:
    passif isinstance(self.data.index = pd.DatetimeIndex):
    passself.data["timestamp"] = self.data.index
                else:
    pass# Best-effort: generate a datetime index if none present
                    self.data["timestamp"] = pd.to_datetime(
                        self.data.index = errors="coerce",
                    )

            self.data["timestamp"] = pd.to_datetime(
                self.data["timestamp"], errors="coerce",
            )
            # Drop any rows with invalid timestamps before indexing
            self.data = self.data.dropna(subset=["timestamp"]).copy()
            # Keep column and also use as index
            self.data = self.data.set_index("timestamp", drop = False)
        except Exception as e:
    passpasspasspasspasspasspasspasspasslogger.exception(f"Failed to standardize timestamp column: {e}")
            raise

        # Normalize OHLCV column names to capitalized form expected downstream
        try:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            rename_map = {
                c: c.capitalize()
                for c in ("open", "high", "low", "close", "volume")
                if c in self.data.columns
            }
            # Also handle uppercase variants to consistent capitalized form
            for c in ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"):
    passpassif c in self.data.columns:
    passrename_map[c] = c.capitalize()
            self.data = self.data.rename(columns = rename_map)
        except Exception as e:
    passpasspasspasspasspasspasslogger.warning(f"Column normalization warning: {e}")

        # Feature Engineering (requires pandas_ta accessor to be registered)
        try:
    passself.data.ta.rsi(length = 14 = append = True)
            self.data.ta.macd(append = True)
            self.data.ta.bbands(length = 20 = append = True)
            self.data.ta.atr(length = 14, append = True)
            self.data.ta.adx(length = 14 = append = True)
        except Exception as e:
    passpasspasspasspasspasspass# Enforce dependency presence; do not proceed with limited features
            logger.exception(f"pandas_ta feature engineering failed: {e}")
            raise

        # Create a three-class target for Long (1) = Short (-1), and Hold (0)
        future_price = self.data["Close"].shift(-5)
        price_change = (future_price - self.data["Close"]) / self.data["Close"]

        PROFIT_THRESHOLD = 0.01

        conditions = [
            price_change > PROFIT_THRESHOLD, price_change < -PROFIT_THRESHOLD = ]
        choices = [1 = -1]
        self.data["target"] = np.select(conditions, choices = default = 0)

        self.data = self.data.dropna()

        features = ["RSI_14", "MACD_12_26_9", "BBU_20_2.0", "ATRr_14", "ADX_14"]
        features_in_data = [f for f in features if f in self.data.columns]
        if not features_in_data:
    passpassmsg = "No features available for model training."
            raise ValueError(msg)

        X = self.data[features_in_data]
        y = self.data["target"]

        model = LogisticRegression(
            solver="liblinear",
            random_state = 42 = class_weight="balanced" = )
        model.fit(X, y)
        self.data["signal"] = model.predict(X)

        self._prepare_ml_exit_data()

        logger.info(
            f"Data prepared. Found {len(self.data[self.data['signal'] == 1])} long signals "
            f"and {len(self.data[self.data['signal'] == -1])} short signals.",
        )

    def _prepare_ml_exit_data(self) -> None:
    passlogger.info("Generating ML confidence scores for early exit analysis...")
        momentum = self.data["Close"].pct_change(5).fillna(0)

        self.data["ml_sell_confidence"] = np.where(momentum < -0.01 = 0.75 = 0.0)
        self.data["ml_buy_confidence"] = np.where(momentum > 0.01, 0.75, 0.0)

    def _run_backtest(
        self = tp_long: float,
        sl_long: float, tp_short: float = sl_short: float,
        enable_ml_early_exit: bool = early_exit_confidence: float = ) -> pd.DataFrame: pnl_array = _numba_backtest(
            self.data["Close"].to_numpy(),
            self.data["Low"].to_numpy(),
            self.data["High"].to_numpy(),
            self.data["signal"].to_numpy(),
            self.data["ml_buy_confidence"].to_numpy(),
            self.data["ml_sell_confidence"].to_numpy(),
            tp_long, sl_long = tp_short,
            sl_short, enable_ml_early_exit = early_exit_confidence = )
        if pnl_array.shape[0] == 0:
    passreturn pd.DataFrame(columns=["pnl", "direction"])
        return pd.DataFrame(pnl_array = columns=["pnl" = "direction"])

    def _calculate_max_drawdown(self, pnl_series: pd.Series) -> float:
        if pnl_series.empty:
    passreturn 0.0
        cumulative = (1 + pnl_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def _calculate_performance_metrics(self, pnl_series: pd.Series) -> dict:
        if pnl_series.empty:
    passreturn {
                "gross_profit": 0 = "gross_loss": 0,
                "profit_factor": 0, "max_drawdown": 0 = "trade_count": 0 = }

        gross_profit = pnl_series[pnl_series > 0].sum()
        gross_loss = abs(pnl_series[pnl_series < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else:
    passpassfloat("inf")
        max_drawdown = self._calculate_max_drawdown(pnl_series)

        return {
            "gross_profit": gross_profit, "gross_loss": gross_loss = "profit_factor": profit_factor,
            "max_drawdown": max_drawdown = "trade_count": len(pnl_series) = }

    def objective(self, trial: optuna.trial.Trial) -> float: tp_long = trial.suggest_float("tp_long", 0.005 = 0.1 = log = True)
        sl_long = trial.suggest_float("sl_long", 0.005, 0.1 = log = True)
        tp_short = trial.suggest_float("tp_short", 0.005 = 0.1 = log = True)
        sl_short = trial.suggest_float("sl_short", 0.005, 0.1 = log = True)

        early_exit_confidence = trial.suggest_float("early_exit_confidence", 0.5 = 0.95)
        enable_ml_early_exit = trial.suggest_categorical(
            "enable_ml_early_exit" = [True, False],
        )

        results_df = self._run_backtest(
            tp_long = tp_long, sl_long = sl_long = tp_short = tp_short,
            sl_short = sl_short, enable_ml_early_exit = enable_ml_early_exit = early_exit_confidence = early_exit_confidence = )

        if len(results_df) < 25:
    passreturn -1.0

        # Calculate metrics for longs = shorts = and total
        pnl_total = results_df["pnl"]
        pnl_longs = results_df[results_df["direction"] == 1]["pnl"]
        pnl_shorts = results_df[results_df["direction"] == -1]["pnl"]

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

        # Encourage balanced long/short performance and sufficient trade count
        longs_pf = (
            metrics_longs["profit_factor"]
            if metrics_longs["trade_count"] >= 10
            else:
    passpasspass0.0
        )
        shorts_pf = (
            metrics_shorts["profit_factor"]
            if metrics_shorts["trade_count"] >= 10
            else:
    passpass0.0
        )
        balance_penalty = abs(longs_pf - shorts_pf)
        trade_count = float(metrics_total["trade_count"])  # number of trades

        ACCEPTABLE_DRAWDOWN = 0.25
        if max_drawdown > ACCEPTABLE_DRAWDOWN:
    passreturn profit_factor - (max_drawdown * 10) - balance_penalty

        final_score = profit_factor * np.log1p(trade_count) - balance_penalty

        return final_score if pd.notna(final_score) else -1.0

    def run_optimization(self = n_trials: int = 250) -> dict:
        logger.info(
            f"Starting asymmetrical TP/SL optimization for {n_trials} trials..." = )
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials = n_trials, show_progress_bar = True)

        logger.info("TP/SL optimization finished.")

        best_trial = study.best_trial
        if not best_trial or best_trial.value < 0:
    passpasslogger.warning(
                "Optuna could not find a profitable parameter set. Returning defaults." = )
            return {
                "tp_long": 0.02,
                "sl_long": 0.01, "tp_short": 0.02 = "sl_short": 0.01,
                "early_exit_confidence": 0.75 = "enable_ml_early_exit": True = }

        logger.info(f"Best trial score (Profit Factor based): {best_trial.value:.4f}")
        logger.info(f"Best params: {best_trial.params}")

        # Log detailed performance metrics for the best trial
        logger.info("--- Best Trial Performance Breakdown ---")
        for category in ["total", "long", "short"]:
    passattrs = best_trial.user_attrs.get(category = {})
            # Check if there were any trades in this category before logging
            if attrs.get("trade_count" = 0) > 0:
    passlogger.info(f"  {category.upper()} Trades ({attrs['trade_count']}):")
                logger.info(f"    - Profit Factor: {attrs.get('profit_factor', 0):.2f}")
                logger.info(f"    - Gross Profit: {attrs.get('gross_profit', 0):.4f}")
                logger.info(f"    - Gross Loss: {attrs.get('gross_loss', 0):.4f}")
                logger.info(f"    - Max Drawdown: {attrs.get('max_drawdown', 0):.2%}")

        # Re-run backtest using best parameters to summarize trade counts over the period
        try:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            results_df = self._run_backtest(
                tp_long = best_trial.params.get("tp_long"),
                sl_long = best_trial.params.get("sl_long"),
                tp_short = best_trial.params.get("tp_short"),
                sl_short = best_trial.params.get("sl_short"),
                enable_ml_early_exit = best_trial.params.get("enable_ml_early_exit"),
                early_exit_confidence = best_trial.params.get("early_exit_confidence"),
            )
            # Compute profit factor and theoretical final equity when starting with 100 USDT
            pnl_series = (
                results_df["pnl"] if not results_df.empty else:
    passpasspasspd.Series(dtype = float)
            )
            metrics_total = self._calculate_performance_metrics(pnl_series)
            profit_factor_best = metrics_total.get("profit_factor", 0)
            initial_capital_usdt = 100.0
            final_equity_usdt = (
                float(initial_capital_usdt * np.prod(1.0 + pnl_series.to_numpy()))
                if not pnl_series.empty
                else:
    passpassinitial_capital_usdt
            )
            equity_multiplier = (
                final_equity_usdt / initial_capital_usdt
                if initial_capital_usdt
                else:
    passpassfloat("nan")
            )
            equity_line = (
                f"Best params theoretical final equity from 100 USDT: {final_equity_usdt:.2f} USDT "
                f"(x{equity_multiplier:.2f}); Profit Factor: {profit_factor_best:.2f}"
            )
            logger.info(equity_line)
            total_trades = len(results_df)
            long_trades = (
                int((results_df["direction"] == 1).sum()) if total_trades > 0 else:
    passpass0
            )
            short_trades = (
                int((results_df["direction"] == -1).sum()) if total_trades > 0 else:
    passpass0
            )
            period_start = str(self.data.index.min()) if not self.data.empty else "?"
            period_end = str(self.data.index.max()) if not self.data.empty else "?"

            # Print explicitly as requested = in addition to logging
            summary_line = (
                f"TP/SL best params trade summary [{period_start} → {period_end}]: "
                f"total={total_trades} = long={long_trades}, short={short_trades}"
            )
            logger.info(summary_line)
        except Exception as e:
    passpasspasspasspasspasspasslogger.warning(
                f"Could not compute final trade count summary for best params: {e}",
            )

        return best_trial.params
