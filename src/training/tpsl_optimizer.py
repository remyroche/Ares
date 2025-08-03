# src/training/tpsl_optimizer.py

import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.analyst.predictive_ensembles.ensemble_orchestrator import (
    RegimePredictiveEnsembles,
)

# Import the existing TechnicalAnalyzer to use its methods directly
from src.analyst.technical_analyzer import TechnicalAnalyzer
from src.database.sqlite_manager import SQLiteManager
from src.utils.logger import logger

# Suppress Optuna's informational messages for a cleaner log during optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TpSlOptimizer:
    """
    Optimizes Take Profit (TP) and Stop Loss (SL) thresholds using Optuna.
    Now includes ML-based early exit capabilities for adverse movement prediction.

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
        self.ml_predictions = None
        self.ensemble_predictor = None
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
            "Preparing data and generating baseline signals for TP/SL optimization...",
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
                f"Could not add order book features using TechnicalAnalyzer: {e}",
            )

        # 3. Initialize ML ensemble predictor for adverse movement detection
        self._initialize_ml_ensemble()

        # 4. Create a simple binary target for the baseline model
        # Use lowercase column names since that's what the data has
        close_col = "close" if "close" in self.data.columns else "Close"
        self.data["target"] = (
            self.data[close_col].shift(-5) > self.data[close_col] * 1.01
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
            f"Using the following features for baseline model: {features_in_data}",
        )

        target = "target"
        X = self.data[features_in_data]
        y = self.data[target]

        if X.empty:
            raise ValueError("Feature set is empty. Cannot train baseline model.")

        # 5. Train a simple model (Logistic Regression) to generate signals
        model = LogisticRegression(
            solver="liblinear",
            random_state=42,
            class_weight="balanced",
            max_iter=1000,
        )
        model.fit(X, y)

        # 6. Generate signals (1 for long, 0 for hold)
        predictions = model.predict(X)
        self.signals = pd.Series(predictions, index=X.index)
        self.data["signal"] = self.signals

        # 7. Generate ML predictions for adverse movement detection
        self._generate_ml_predictions()

        logger.info(
            f"Baseline signals generated. Found {len(self.signals[self.signals == 1])} potential long signals.",
        )

    def _initialize_ml_ensemble(self):
        """Initialize ML ensemble for adverse movement prediction."""
        try:
            # Initialize ensemble predictor
            config = {
                "analyst": {
                    "global_meta_learner": {
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "max_depth": 6,
                        "random_state": 42,
                    },
                },
            }
            self.ensemble_predictor = RegimePredictiveEnsembles(config)
            logger.info("ML ensemble predictor initialized successfully")
        except Exception as e:
            logger.warning(
                f"Could not initialize ML ensemble: {e}. Will use fallback method.",
            )
            self.ensemble_predictor = None

    def _generate_ml_predictions(self):
        """Generate ML predictions for adverse movement detection."""
        if self.ensemble_predictor is None:
            logger.warning("ML ensemble not available. Using fallback predictions.")
            # Fallback: simple momentum-based predictions
            self.ml_predictions = self._generate_fallback_predictions()
            return

        try:
            # Prepare features for ML prediction
            ml_features = self._prepare_ml_features()

            # Get predictions from ensemble
            predictions = self.ensemble_predictor.get_all_predictions(
                asset=self.symbol,
                current_features=ml_features,
            )

            # Store predictions
            self.ml_predictions = predictions
            logger.info("ML predictions generated successfully")

        except Exception as e:
            logger.error(f"Error generating ML predictions: {e}")
            self.ml_predictions = self._generate_fallback_predictions()

    def _prepare_ml_features(self) -> pd.DataFrame:
        """Prepare features for ML ensemble prediction."""
        # Select relevant features for ML prediction
        feature_columns = [
            "RSI_14",
            "SMA_50",
            "MACD_12_26_9",
            "MACDs_12_26_9",
            "MACDh_12_26_9",
            "ATRr_14",
            "ADX_14",
            "OBV",
            "VWAP",
        ]

        # Filter available features
        available_features = [
            col for col in feature_columns if col in self.data.columns
        ]

        if not available_features:
            # Fallback to basic features
            available_features = ["Close", "Volume"]

        return self.data[available_features].fillna(0)

    def _generate_fallback_predictions(self) -> dict:
        """Generate fallback predictions when ML ensemble is not available."""
        # Simple momentum-based predictions
        close_prices = (
            self.data["Close"] if "Close" in self.data.columns else self.data["close"]
        )

        # Calculate momentum
        momentum = close_prices.pct_change(5)

        # Generate predictions based on momentum
        predictions = []
        for i in range(len(momentum)):
            if pd.isna(momentum.iloc[i]):
                predictions.append({"prediction": "HOLD", "confidence": 0.5})
            elif momentum.iloc[i] > 0.01:  # 1% positive momentum
                predictions.append({"prediction": "BUY", "confidence": 0.7})
            elif momentum.iloc[i] < -0.01:  # 1% negative momentum
                predictions.append({"prediction": "SELL", "confidence": 0.7})
            else:
                predictions.append({"prediction": "HOLD", "confidence": 0.5})

        return {
            "predictions": predictions,
            "final_prediction": "HOLD",
            "final_confidence": 0.5,
        }

    def _check_adverse_movement(
        self,
        position_direction: str,
        current_prediction: dict,
        confidence_threshold: float = 0.6,
    ) -> bool:
        """
        Check if ML predicts adverse movement that warrants early exit.

        Args:
            position_direction: "LONG" or "SHORT"
            current_prediction: ML prediction dict
            confidence_threshold: Minimum confidence for early exit

        Returns:
            bool: True if adverse movement detected
        """
        if not current_prediction:
            return False

        prediction = current_prediction.get("prediction", "HOLD")
        confidence = current_prediction.get("confidence", 0.0)

        # Check for adverse movement
        if (
            position_direction == "LONG"
            and prediction == "SELL"
            or position_direction == "SHORT"
            and prediction == "BUY"
        ):
            if confidence > confidence_threshold:
                return True

        return False

    def _run_backtest(
        self,
        tp_threshold: float,
        sl_threshold: float,
        early_exit_confidence: float = 0.6,
        enable_ml_early_exit: bool = True,
    ) -> pd.Series:
        """
        Runs a simplified iterative backtest based on pre-generated signals.
        Now includes ML-based early exit for adverse movement.
        """
        pnls = []
        position_open = False
        entry_price = 0.0
        position_direction = "LONG"  # Default for long-only strategy

        # Use lowercase column names since that's what the data has
        close_col = "close" if "close" in self.data.columns else "Close"
        low_col = "low" if "low" in self.data.columns else "Low"
        high_col = "high" if "high" in self.data.columns else "High"

        close_prices = self.data[close_col].to_numpy()
        low_prices = self.data[low_col].to_numpy()
        high_prices = self.data[high_col].to_numpy()
        signals = self.data["signal"].to_numpy()

        for i in range(1, len(self.data)):
            if not position_open and signals[i - 1] == 1:
                position_open = True
                entry_price = close_prices[i - 1]

            if position_open:
                # Check for ML-based early exit if enabled
                if enable_ml_early_exit and self.ml_predictions:
                    current_prediction = self._get_current_ml_prediction(i)
                    if self._check_adverse_movement(
                        position_direction,
                        current_prediction,
                        early_exit_confidence,
                    ):
                        # Early exit due to adverse ML prediction
                        current_price = close_prices[i]
                        pnl = (current_price - entry_price) / entry_price
                        pnls.append(pnl)
                        position_open = False
                        continue

                # Check traditional TP/SL
                if high_prices[i] >= entry_price * (1 + tp_threshold):
                    pnls.append(tp_threshold)
                    position_open = False
                elif low_prices[i] <= entry_price * (1 - sl_threshold):
                    pnls.append(-sl_threshold)
                    position_open = False

        return pd.Series(pnls)

    def _get_current_ml_prediction(self, index: int) -> dict:
        """Get ML prediction for current index."""
        if not self.ml_predictions:
            return {"prediction": "HOLD", "confidence": 0.5}

        try:
            if "predictions" in self.ml_predictions:
                # Fallback predictions
                if index < len(self.ml_predictions["predictions"]):
                    return self.ml_predictions["predictions"][index]
            else:
                # Ensemble predictions
                return {
                    "prediction": self.ml_predictions.get("final_prediction", "HOLD"),
                    "confidence": self.ml_predictions.get("final_confidence", 0.5),
                }
        except Exception as e:
            logger.warning(f"Error getting ML prediction: {e}")

        return {"prediction": "HOLD", "confidence": 0.5}

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        The objective function for Optuna to maximize.
        Now includes ML early exit parameters.
        """
        # Traditional TP/SL parameters
        tp = trial.suggest_float("tp_threshold", 0.002, 0.05, log=True)
        sl = trial.suggest_float("sl_threshold", 0.002, 0.05, log=True)

        # ML early exit parameters
        early_exit_confidence = trial.suggest_float("early_exit_confidence", 0.5, 0.9)
        enable_ml_early_exit = trial.suggest_categorical(
            "enable_ml_early_exit",
            [True, False],
        )

        # Risk management parameters
        max_drawdown_threshold = trial.suggest_float(
            "max_drawdown_threshold",
            0.05,
            0.20,
        )
        max_daily_loss = trial.suggest_float("max_daily_loss", 0.02, 0.10)

        pnl_series = self._run_backtest(
            tp_threshold=tp,
            sl_threshold=sl,
            early_exit_confidence=early_exit_confidence,
            enable_ml_early_exit=enable_ml_early_exit,
        )

        if len(pnl_series) < 20:
            return -1.0

        # Calculate performance metrics
        std_dev = pnl_series.std()
        if std_dev < 1e-9:
            return 0.0

        sharpe_ratio = pnl_series.mean() / std_dev

        # Apply risk management penalties
        max_drawdown = self._calculate_max_drawdown(pnl_series)
        daily_loss = self._calculate_max_daily_loss(pnl_series)

        # Penalize if risk thresholds are exceeded
        if max_drawdown > max_drawdown_threshold:
            sharpe_ratio *= 0.5  # 50% penalty for exceeding drawdown

        if daily_loss > max_daily_loss:
            sharpe_ratio *= 0.7  # 30% penalty for exceeding daily loss

        return sharpe_ratio if pd.notna(sharpe_ratio) else -1.0

    def _calculate_max_drawdown(self, pnl_series: pd.Series) -> float:
        """Calculate maximum drawdown from PnL series."""
        cumulative = (1 + pnl_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def _calculate_max_daily_loss(self, pnl_series: pd.Series) -> float:
        """Calculate maximum daily loss from PnL series."""
        # Group by day (assuming hourly data)
        daily_pnl = pnl_series.groupby(pnl_series.index.date).sum()
        return abs(daily_pnl.min())

    def run_optimization(self, n_trials: int = 150) -> dict:
        """
        Executes the Optuna optimization study.
        Now includes ML early exit and risk management parameters.
        """
        logger.info(
            f"Starting TP/SL optimization with ML early exit for {n_trials} trials...",
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        if not study.best_trial or study.best_value < 0:
            logger.warning(
                "Optuna study did not find a profitable set of TP/SL parameters.",
            )
            return {
                "tp_threshold": 0.01,
                "sl_threshold": 0.01,
                "early_exit_confidence": 0.7,
                "enable_ml_early_exit": True,
                "max_drawdown_threshold": 0.10,
                "max_daily_loss": 0.05,
            }

        logger.info("TP/SL optimization finished.")
        logger.info(f"Best trial value (Sharpe Ratio): {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params

    def get_optimization_summary(self) -> dict:
        """
        Get a summary of the optimization results including ML early exit statistics.
        """
        return {
            "optimization_completed": True,
            "ml_ensemble_available": self.ensemble_predictor is not None,
            "total_signals": len(self.signals) if self.signals is not None else 0,
            "ml_predictions_generated": self.ml_predictions is not None,
            "recommended_parameters": self.run_optimization(),
            "risk_management_rules": {
                "max_position_size": 0.1,
                "max_daily_loss": 0.05,
                "max_drawdown": 0.15,
                "kill_switch_threshold": 0.10,
            },
        }
