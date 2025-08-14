# src/analyst/unified_regime_classifier.py
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import CONFIG
from src.utils.logger import system_logger
from src.utils.error_handler import (
    handle_errors,
    create_fallback_strategy,
    handle_data_processing_errors,
    handle_type_conversions,
    safe_division,
    clean_dataframe,
)
from src.utils.warning_symbols import (
    error,
    failed,
    warning,
)


class UnifiedRegimeClassifier:
    """
    Unified Market Regime Classifier with HMM-based labeling and ensemble prediction.

    Approach:
    1. HMM-based labeling for basic regimes (BULL, BEAR, SIDEWAYS)
    2. Ensemble prediction with majority voting for basic regimes
    3. Location classification (SUPPORT, RESISTANCE, OPEN_RANGE)

    Enhanced with:
    - Comprehensive logging for troubleshooting and efficiency monitoring
    - Thorough error handling using decorators
    - Proper data usage (scaling, normalization, returns vs prices)
    - Complete type hints throughout
    """

    def __init__(
        self,
        config: Dict[str, Any],
        exchange: str = "UNKNOWN",
        symbol: str = "UNKNOWN",
    ) -> None:
        """
        Initialize the UnifiedRegimeClassifier.

        Args:
            config: Configuration dictionary
            exchange: Exchange name
            symbol: Trading symbol
        """
        # Ensure NumPy RNG pickles created under different versions can be loaded
        self._enable_numpy_rng_unpickle_compat(system_logger)
        
        # Configuration setup with enhanced logging
        self.config = config.get("analyst", {}).get("unified_regime_classifier", {})
        self.global_config = config
        self.logger = system_logger.getChild("UnifiedRegimeClassifier")
        self.exchange = exchange
        self.symbol = symbol
        
        # Add print method for compatibility
        self.print = self.logger.info

        # HMM Configuration - enforce at least 3 states (BULL, BEAR, SIDEWAYS)
        configured_states = int(self.config.get("n_states", 3))
        self.n_states = max(3, configured_states)
        self.n_iter = self.config.get("n_iter", 100)
        self.random_state = self.config.get("random_state", 42)
        self.target_timeframe = self.config.get("target_timeframe", "1h")
        self.volatility_period = self.config.get("volatility_period", 10)

        # Enhanced logging of configuration
        self.logger.info(
            {
                "msg": "UnifiedRegimeClassifier configuration loaded",
                "exchange": self.exchange,
                "symbol": self.symbol,
                "n_states_configured": configured_states,
                "n_states_enforced_min": self.n_states,
                "target_timeframe": self.target_timeframe,
                "volatility_period": self.volatility_period,
                "min_data_points_default": self.config.get("min_data_points", 1000),
            }
        )

        # Thresholds for regime interpretation (configurable)
        # Optimized thresholds for better regime balance
        self.adx_sideways_threshold = self.config.get("adx_sideways_threshold", 18)
        self.volatility_threshold = self.config.get("volatility_threshold", 0.025)
        self.atr_normalized_threshold = self.config.get("atr_normalized_threshold", 0.035)
        self.volatility_percentile_threshold = self.config.get("volatility_percentile_threshold", 0.80)
        self.bb_width_volatility_threshold = self.config.get("bb_width_volatility_threshold", 0.045)

        # Log threshold configuration
        self.logger.info(
            {
                "msg": "Regime classification thresholds",
                "thresholds": {
                    "adx_sideways_threshold": self.adx_sideways_threshold,
                    "volatility_threshold": self.volatility_threshold,
                    "atr_normalized_threshold": self.atr_normalized_threshold,
                    "volatility_percentile_threshold": self.volatility_percentile_threshold,
                    "bb_width_volatility_threshold": self.bb_width_volatility_threshold,
                },
            }
        )

        # Detect BLANK mode and adjust minimum data points accordingly
        blank_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
        if blank_mode:
            self.min_data_points = self.config.get("min_data_points", 50)
            self.logger.info("🔧 BLANK MODE DETECTED: Using reduced minimum data points (50)")
        else:
            self.min_data_points = self.config.get("min_data_points", 1000)

        # Models with type hints
        self.hmm_model: Optional[hmm.GaussianHMM] = None
        self.scaler: Optional[StandardScaler] = None
        self.state_to_regime_map: Dict[int, str] = {}

        # Ensemble Models for Basic Regimes
        self.basic_ensemble: Optional[LGBMClassifier] = None

        # Location Classifier
        self.location_classifier: Optional[LGBMClassifier] = None
        self.location_label_encoder: Optional[LabelEncoder] = None

        # Legacy S/R/Candle code removed
        self.enable_sr_integration = self.config.get("enable_sr_integration", True)
        self.basic_label_encoder: Optional[LabelEncoder] = None

        # Training Status
        self.trained: bool = False
        self.last_training_time: Optional[datetime] = None

        # Model Paths with enhanced path resolution
        self._setup_model_paths()

        self.logger.info("✅ UnifiedRegimeClassifier initialized successfully")

    def _setup_model_paths(self) -> None:
        """Setup model paths with enhanced error handling and logging."""
        try:
            # Resolve checkpoints directory to an absolute path anchored at the project root
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            base_checkpoint_dir = CONFIG.get("CHECKPOINT_DIR", "checkpoints")
            if not os.path.isabs(base_checkpoint_dir):
                base_checkpoint_dir = os.path.join(project_root, base_checkpoint_dir)

            self.model_dir = os.path.join(base_checkpoint_dir, "analyst_models")
            
            # Optional hierarchical directory for compatibility
            self._hierarchical_model_dir = os.path.join(
                self.model_dir,
                self.exchange,
                self.symbol,
                self.target_timeframe,
            )
            
            # Create model directory
            os.makedirs(self.model_dir, exist_ok=True)

            # Model file paths
            self.hmm_model_path = os.path.join(
                self.model_dir,
                f"unified_hmm_model_{self.exchange}_{self.symbol}_{self.target_timeframe}.joblib",
            )
            self.ensemble_model_path = os.path.join(
                self.model_dir,
                f"unified_ensemble_model_{self.exchange}_{self.symbol}_{self.target_timeframe}.joblib",
            )
            self.location_model_path = os.path.join(
                self.model_dir,
                f"unified_location_model_{self.exchange}_{self.symbol}_{self.target_timeframe}.joblib",
            )

            self.logger.info(
                {
                    "msg": "Model paths configured",
                    "model_dir": self.model_dir,
                    "hmm_model_path": self.hmm_model_path,
                    "ensemble_model_path": self.ensemble_model_path,
                    "location_model_path": self.location_model_path,
                }
            )

        except Exception as e:
            self.logger.error(f"❌ Failed to setup model paths: {e}")
            raise

    # --- Compatibility shim for NumPy RNG unpickling across versions ---
    _NUMPY_RNG_UNPICKLE_PATCHED = False

    @staticmethod
    def _enable_numpy_rng_unpickle_compat(logger: Optional[Any] = None) -> None:
        """Enable compatibility for unpickling NumPy RNG BitGenerators."""
        if getattr(
            UnifiedRegimeClassifier._enable_numpy_rng_unpickle_compat, "_patched", False
        ):
            return
        try:
            import numpy.random._pickle as np_random_pickle  # type: ignore[attr-defined]

            original_ctor = getattr(np_random_pickle, "__bit_generator_ctor", None)
            if original_ctor is None:
                UnifiedRegimeClassifier._enable_numpy_rng_unpickle_compat._patched = True
                return

            def _normalized_numpy_bitgen_ctor(
                bit_generator_name: Any, state: Optional[Any] = None, *args: Any, **kwargs: Any
            ) -> Any:
                name_candidate = bit_generator_name
                try:
                    if hasattr(name_candidate, "__name__"):
                        name_candidate = name_candidate.__name__
                    elif isinstance(name_candidate, str) and name_candidate.startswith("<class "):
                        name_candidate = name_candidate.split(".")[-1].split("'>")[0]
                except Exception:
                    pass
                effective_state = kwargs.get("state", state)
                try:
                    return original_ctor(name_candidate, effective_state)
                except (TypeError, ValueError):
                    try:
                        return original_ctor(name_candidate)
                    except Exception as ctor_exc:
                        try:
                            import numpy as _np
                            bitgen_cls = getattr(_np.random, name_candidate, None)
                            if bitgen_cls is None and name_candidate == "MT19937":
                                try:
                                    import numpy.random._mt19937 as _mt  # type: ignore[attr-defined]
                                    bitgen_cls = getattr(_mt, "MT19937", None)
                                except Exception:
                                    bitgen_cls = None
                            if bitgen_cls is not None:
                                return bitgen_cls()
                        except Exception:
                            pass
                        raise ctor_exc

            np_random_pickle.__bit_generator_ctor = _normalized_numpy_bitgen_ctor  # type: ignore[attr-defined]
            UnifiedRegimeClassifier._enable_numpy_rng_unpickle_compat._patched = True
            if logger is not None:
                logger.info("Applied NumPy RNG unpickle compatibility shim (URC)")
        except Exception as _shim_exc:
            UnifiedRegimeClassifier._enable_numpy_rng_unpickle_compat._patched = True
            if logger is not None:
                logger.warning(
                    warning(f"NumPy RNG unpickle shim not applied (URC): {_shim_exc}")
                )

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="UnifiedRegimeClassifier.initialize",
    )
    async def initialize(self) -> bool:
        """
        Initialize the UnifiedRegimeClassifier.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.logger.info(
            f"Initializing UnifiedRegimeClassifier for {self.exchange}_{self.symbol}",
        )

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

        # Try to load existing models
        if self.load_models():
            self.logger.info("✅ Loaded existing models successfully")
            self.trained = True
        else:
            self.logger.info("ℹ️  No existing models found, will train new models")
            self.trained = False

        self.logger.info("✅ UnifiedRegimeClassifier initialized successfully")
        return True

    @handle_data_processing_errors(
        default_return=pd.DataFrame(),
        context="UnifiedRegimeClassifier._calculate_features",
    )
    def _calculate_features(
        self,
        klines_df: pd.DataFrame,
        min_data_points: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Calculate comprehensive features for regime and location classification.

        Args:
            klines_df: DataFrame with OHLCV data
            min_data_points: Minimum data points required

        Returns:
            DataFrame with calculated features
        """
        if min_data_points is None:
            min_data_points = self.min_data_points

        # Enhanced data validation and logging
        self.logger.info(
            {
                "msg": "Starting feature calculation",
                "input_rows": len(klines_df),
                "min_data_points": min_data_points,
                "columns": list(klines_df.columns),
            }
        )

        if len(klines_df) < min_data_points:
            self.logger.warning(
                f"Insufficient data: {len(klines_df)} < {min_data_points}. Consider reducing min_data_points or collecting more data.",
            )
            # Try with a lower threshold if possible
            if len(klines_df) >= 200:  # Minimum viable amount
                self.logger.info(
                    f"Proceeding with {len(klines_df)} data points (reduced from {min_data_points})",
                )
                min_data_points = len(klines_df)
            else:
                self.logger.error(
                    f"Data too small: {len(klines_df)} < 200 minimum required",
                )
                return pd.DataFrame()

        self.logger.info(f"🔧 Calculating features for {len(klines_df)} periods...")

        # Create features DataFrame with proper data cleaning
        features_df = clean_dataframe(klines_df.copy())

        # Basic price features using price differences (returns-based approach)
        features_df["log_returns"] = np.log(
            safe_division(features_df["close"], features_df["close"].shift(1))
        )
        features_df["price_change"] = features_df["close"].pct_change()
        
        # Enhanced ratio calculations with better handling of edge cases
        high_diff = features_df["high"].diff()
        low_diff = features_df["low"].diff()
        open_diff = features_df["open"].diff()
        close_diff = features_df["close"].diff()
        
        # Use safer division with clipping to prevent extreme values
        features_df["high_low_diff_ratio"] = np.where(
            np.abs(low_diff) > 1e-8,
            np.clip(safe_division(high_diff, low_diff), -100, 100),
            1.0  # Default to neutral ratio when denominator is too small
        )
        
        features_df["close_open_diff_ratio"] = np.where(
            np.abs(open_diff) > 1e-8,
            np.clip(safe_division(close_diff, open_diff), -100, 100),
            1.0  # Default to neutral ratio when denominator is too small
        )

        # Volatility features with enhanced calculation
        features_df["volatility_20"] = features_df["log_returns"].rolling(20).std()
        features_df["volatility_10"] = features_df["log_returns"].rolling(10).std()
        features_df["volatility_5"] = features_df["log_returns"].rolling(5).std()

        # EWMA-based volatility provides a smoother, more reactive estimate
        features_df["ewma_volatility_20"] = (
            features_df["log_returns"].ewm(span=20, adjust=False).std()
        )

        # Volume features with proper normalization
        volume_ma = features_df["volume"].rolling(20).mean()
        features_df["volume_ratio"] = np.where(
            volume_ma > 1e-8,
            np.clip(safe_division(features_df["volume"], volume_ma), 0, 50),
            1.0  # Default to neutral ratio when denominator is too small
        )
        features_df["volume_change"] = features_df["volume"].pct_change()

        # Technical indicators with enhanced error handling
        features_df = self._calculate_rsi(features_df)
        features_df = self._calculate_macd(features_df)
        features_df = self._calculate_bollinger_bands(features_df)
        features_df = self._calculate_atr(features_df)
        
        # Enhanced ATR normalization with clipping
        atr_denominator = features_df["close"].diff().abs() + 1e-8
        features_df["atr_normalized"] = np.clip(
            safe_division(features_df["atr"], atr_denominator), 0, 10
        )
        
        features_df = self._calculate_adx(features_df)

        # Enhanced volatility features for VOLATILE regime detection
        features_df["volatility_regime"] = self._calculate_volatility_regime(features_df)
        features_df["volatility_acceleration"] = features_df["volatility_20"].diff()
        features_df["volatility_momentum"] = (
            features_df["volatility_20"] - features_df["volatility_20"].shift(5)
        )

        # Comprehensive NaN handling with detailed logging
        initial_length = len(features_df)
        features_df = self._handle_nan_values(features_df)
        dropped_rows = initial_length - len(features_df)

        self.logger.info(
            {
                "msg": "Feature calculation completed",
                "final_rows": len(features_df),
                "dropped_rows": dropped_rows,
                "feature_columns": list(features_df.columns),
                "data_types": features_df.dtypes.to_dict(),
            }
        )

        return features_df

    def _handle_nan_values(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NaN values with comprehensive strategy and logging.
        
        Args:
            features_df: DataFrame with potential NaN values
            
        Returns:
            DataFrame with NaN values handled
        """
        # Log initial NaN counts
        nan_counts = features_df.isnull().sum()
        if nan_counts.sum() > 0:
            self.logger.info(f"🔍 Initial NaN counts: {nan_counts[nan_counts > 0].to_dict()}")

        # Improved NaN handling: use forward fill for technical indicators
        technical_columns = [
            "rsi", "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_position", "bb_width",
            "atr", "atr_normalized", "adx", "volatility_regime",
        ]

        for col in technical_columns:
            if col in features_df.columns:
                # Forward fill NaN values for technical indicators
                features_df[col] = features_df[col].ffill()
                # Fill any remaining NaN values with 0
                features_df[col] = features_df[col].fillna(0)

        # For log_returns and other price-based features, use 0 for NaN
        price_features = [
            "log_returns", "price_change", "volume_change",
            "volatility_acceleration", "volatility_momentum",
        ]
        for col in price_features:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0)

        # For ratio features, use 1 for NaN (neutral ratio)
        ratio_features = [
            "high_low_diff_ratio", "close_open_diff_ratio", "volume_ratio",
        ]
        for col in ratio_features:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(1)

        # For volatility features, use 0 for NaN
        vol_features = [
            "volatility_20", "volatility_10", "volatility_5", "ewma_volatility_20",
        ]
        for col in vol_features:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0)

        # CRITICAL: Handle infinity and extremely large values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Clip extremely large values to prevent HMM training issues
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in features_df.columns:
                # Clip to reasonable ranges based on feature type
                if 'ratio' in col.lower():
                    features_df[col] = np.clip(features_df[col], -100, 100)
                elif 'volatility' in col.lower():
                    features_df[col] = np.clip(features_df[col], 0, 1)
                elif 'volume' in col.lower():
                    features_df[col] = np.clip(features_df[col], 0, 100)
                elif 'rsi' in col.lower():
                    features_df[col] = np.clip(features_df[col], 0, 100)
                elif 'macd' in col.lower():
                    features_df[col] = np.clip(features_df[col], -10, 10)
                elif 'bb_' in col.lower():
                    features_df[col] = np.clip(features_df[col], -10, 10)
                elif 'atr' in col.lower():
                    features_df[col] = np.clip(features_df[col], 0, 10)
                elif 'adx' in col.lower():
                    features_df[col] = np.clip(features_df[col], 0, 100)
                else:
                    # Default clipping for other numeric features
                    features_df[col] = np.clip(features_df[col], -1000, 1000)

        # Only drop rows that still have NaN values after all the filling
        final_nan_counts = features_df.isnull().sum()
        if final_nan_counts.sum() > 0:
            self.logger.warning(f"⚠️ Remaining NaN counts after handling: {final_nan_counts[final_nan_counts > 0].to_dict()}")
            features_df = features_df.dropna()

        return features_df

    def _calculate_volatility_regime(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Calculate volatility regime for VOLATILE classification.
        
        Args:
            features_df: DataFrame with calculated features
            
        Returns:
            Series with volatility regime indicators
        """
        # Calculate rolling volatility percentiles (prefer smoothed EWMA if present)
        vol_baseline = (
            features_df["ewma_volatility_20"]
            if "ewma_volatility_20" in features_df.columns
            else features_df["volatility_20"]
        )
        vol_20_percentile = vol_baseline.rolling(100).rank(pct=True)
        vol_10_percentile = features_df["volatility_10"].rolling(100).rank(pct=True)

        # High volatility regime (configurable top percentile of volatility)
        pct = float(self.volatility_percentile_threshold)
        # Use OR to be more permissive (either horizon in the top percentile marks high vol)
        high_vol = (vol_20_percentile > pct) | (vol_10_percentile > pct)

        # Additional breadth indicators of volatility
        atr_norm_high = (
            features_df["atr_normalized"] > float(self.atr_normalized_threshold)
            if "atr_normalized" in features_df.columns
            else False
        )
        bb_width_high = (
            features_df["bb_width"] > float(self.bb_width_volatility_threshold)
            if "bb_width" in features_df.columns
            else False
        )

        # Volatility acceleration (more permissive)
        vol_accel = features_df["volatility_20"].diff() > 0

        # Combine conditions for VOLATILE regime
        volatile_regime = high_vol | atr_norm_high | bb_width_high | vol_accel

        return volatile_regime.astype(int)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="UnifiedRegimeClassifier._calculate_rsi",
    )
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI indicator using price differences."""
        # Use price differences instead of absolute prices
        close_diff = df["close"].diff()
        gain = (close_diff.where(close_diff > 0, 0)).rolling(window=period).mean()
        loss = (-close_diff.where(close_diff < 0, 0)).rolling(window=period).mean()
        rs = safe_division(gain, loss)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="UnifiedRegimeClassifier._calculate_macd",
    )
    def _calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Calculate MACD indicator using price differences."""
        # Use price differences instead of absolute prices
        close_diff = df["close"].diff()
        exp1 = close_diff.ewm(span=fast).mean()
        exp2 = close_diff.ewm(span=slow).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=signal).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        return df

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="UnifiedRegimeClassifier._calculate_adx",
    )
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate the Average Directional Index (ADX)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate True Range (TR)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()

        # Calculate Directional Movement (+DM, -DM)
        move_up = high.diff()
        move_down = low.diff()
        plus_dm = ((move_up > move_down) & (move_up > 0)) * move_up
        minus_dm = ((move_down > move_up) & (move_down > 0)) * move_down

        plus_dm = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
        minus_dm = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

        # Calculate Directional Index (+DI, -DI)
        plus_di = 100 * safe_division(plus_dm, atr)
        minus_di = 100 * safe_division(minus_dm, atr)

        # Calculate Directional Movement Index (DX) and ADX
        dx = 100 * safe_division(abs(plus_di - minus_di), (plus_di + minus_di))
        df["adx"] = dx.ewm(alpha=1 / period, adjust=False).mean()
        df["adx"] = df["adx"].fillna(25)  # Fill initial NaNs with a neutral value

        return df

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="UnifiedRegimeClassifier._calculate_bollinger_bands",
    )
    def _calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2,
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands using price differences."""
        # Use price differences instead of absolute prices
        close_diff = df["close"].diff()
        sma = close_diff.rolling(window=period).mean()
        std = close_diff.rolling(window=period).std()
        df["bb_upper"] = sma + (std * std_dev)
        df["bb_lower"] = sma - (std * std_dev)
        df["bb_position"] = safe_division(
            (close_diff - df["bb_lower"]),
            (df["bb_upper"] - df["bb_lower"])
        )
        df["bb_width"] = safe_division(
            (df["bb_upper"] - df["bb_lower"]), sma
        )
        return df

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="UnifiedRegimeClassifier._calculate_atr",
    )
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range using price differences."""
        # Use price differences instead of absolute prices
        high_diff = df["high"].diff()
        low_diff = df["low"].diff()
        close_diff = df["close"].diff()

        high_low_diff = high_diff - low_diff
        high_close_diff = np.abs(high_diff - close_diff.shift())
        low_close_diff = np.abs(low_diff - close_diff.shift())
        true_range = np.maximum(
            high_low_diff, np.maximum(high_close_diff, low_close_diff)
        )
        df["atr"] = true_range.rolling(window=period).mean()
        return df

    def _interpret_hmm_states(
        self,
        features_df: pd.DataFrame,
        state_sequence: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Interpret HMM states and map them to basic market regimes.
        Now uses simplified logic focusing on directional trends only.
        
        Args:
            features_df: DataFrame with calculated features
            state_sequence: Array of HMM state predictions
            
        Returns:
            Dict containing state analysis and regime mapping
        """
        analysis_df = features_df.copy()
        analysis_df["state"] = state_sequence
        state_analysis: Dict[str, Any] = {}

        # Thresholds for regime classification come from instance configuration
        adx_sideways_threshold = self.adx_sideways_threshold
        volatility_threshold = self.volatility_threshold
        atr_norm_threshold = self.atr_normalized_threshold

        self.logger.info(
            {
                "msg": "Interpreting HMM states",
                "n_states": self.n_states,
                "state_sequence_length": len(state_sequence),
                "unique_states": sorted(np.unique(state_sequence)),
                "thresholds": {
                    "adx_sideways_threshold": adx_sideways_threshold,
                    "volatility_threshold": volatility_threshold,
                    "atr_normalized_threshold": atr_norm_threshold,
                },
            }
        )

        for state in range(self.n_states):
            state_data = analysis_df[analysis_df["state"] == state]

            if len(state_data) == 0:
                continue

            # Calculate state characteristics
            mean_return = state_data["log_returns"].mean()
            mean_volatility = state_data["volatility_20"].mean()
            mean_adx = state_data["adx"].mean()
            mean_atr_norm = state_data["atr_normalized"].mean()

            # Optimized regime classification logic for better balance
            # Check for sideways movement (moderate directional strength)
            is_sideways = mean_adx < adx_sideways_threshold
            
            # Tweak thresholds to reduce SIDEWAYS dominance
            small_return = 0.0002
            dir_return = 0.0004

            # First check if it's clearly sideways (low ADX and very small returns)
            if is_sideways and abs(mean_return) < small_return:
                regime = "SIDEWAYS"
            # Then check for directional movements with slightly lower return threshold
            elif mean_return > dir_return and mean_adx >= adx_sideways_threshold:
                regime = "BULL"
            elif mean_return < -dir_return and mean_adx >= adx_sideways_threshold:
                regime = "BEAR"
            # For borderline cases, use a balanced approach
            elif is_sideways:
                # If ADX is low but returns are modest, still classify as directional
                if abs(mean_return) >= small_return:
                    regime = "BULL" if mean_return > 0 else "BEAR"
                else:
                    regime = "SIDEWAYS"
            else:
                # Default to directional based on return sign
                regime = "BULL" if mean_return >= 0 else "BEAR"

            state_analysis[state] = {
                "regime": regime,
                "mean_return": mean_return,
                "mean_volatility": mean_volatility,
                "mean_adx": mean_adx,
                "mean_atr_norm": mean_atr_norm,
                "count": len(state_data),
            }

            self.logger.info(
                f"State {state}: {regime} "
                f"(mean_return={mean_return:.4f}, mean_vol={mean_volatility:.4f}, mean_adx={mean_adx:.2f}, "
                f"is_sideways={is_sideways})",
            )

        # Create state to regime mapping
        state_to_regime_map = {
            state: data["regime"]
            for state, data in state_analysis.items()
            if isinstance(state, int)
        }

        # Persist mapping without post-hoc coverage enforcement
        state_analysis["state_to_regime_map"] = state_to_regime_map

        # Post-mapping balancing: softly constrain SIDEWAYS to ~20-40% by reassigning borderline states
        try:
            total = sum(d.get("count", 0) for s, d in state_analysis.items() if s != "state_to_regime_map")
            if total > 0:
                sideways_count = sum(d.get("count", 0) for s, d in state_analysis.items() if s != "state_to_regime_map" and d.get("regime") == "SIDEWAYS")
                sideways_ratio = sideways_count / total
                target_min, target_max = 0.20, 0.40
                if sideways_ratio > target_max:
                    # Reassign the weakest SIDEWAYS states (highest |mean_return|) to directional to reduce ratio
                    candidates = [
                        (s, d) for s, d in state_analysis.items()
                        if s != "state_to_regime_map" and d.get("regime") == "SIDEWAYS"
                    ]
                    # Sort by abs(mean_return) descending (more directional), then by mean_adx descending
                    candidates.sort(key=lambda x: (abs(x[1].get("mean_return", 0.0)), x[1].get("mean_adx", 0.0)), reverse=True)
                    to_flip = 0
                    while sideways_ratio > target_max and to_flip < len(candidates):
                        s, d = candidates[to_flip]
                        new_regime = "BULL" if d.get("mean_return", 0.0) >= 0 else "BEAR"
                        state_analysis[s]["regime"] = new_regime
                        sideways_count -= d.get("count", 0)
                        sideways_ratio = sideways_count / total if total else sideways_ratio
                        to_flip += 1
                    # Rebuild map
                    state_to_regime_map = {s: d["regime"] for s, d in state_analysis.items() if s != "state_to_regime_map"}
                    state_analysis["state_to_regime_map"] = state_to_regime_map
        except Exception as e:
            self.logger.warning(f"Error in regime balancing: {e}")

        # Log summary of how regimes are derived from HMM states
        mapped_counts: Dict[str, int] = {}
        for state, data in state_analysis.items():
            if state == "state_to_regime_map":
                continue
            mapped_counts[data["regime"]] = mapped_counts.get(data["regime"], 0) + int(
                data.get("count", 0)
            )

        self.logger.info(
            {
                "msg": "HMM state mapping summary",
                "n_states": self.n_states,
                "unique_mapped_regimes": sorted(
                    list({r for r in state_to_regime_map.values()})
                ),
                "mapped_counts": mapped_counts,
                "thresholds": {
                    "adx_sideways_threshold": adx_sideways_threshold,
                    "volatility_threshold": volatility_threshold,
                    "atr_normalized_threshold": atr_norm_threshold,
                },
            }
        )

        return state_analysis

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="UnifiedRegimeClassifier.train_hmm_labeler",
    )
    async def train_hmm_labeler(self, historical_klines: pd.DataFrame) -> bool:
        """
        Train HMM-based labeler for basic regimes (BULL, BEAR, SIDEWAYS).
        
        Args:
            historical_klines: Historical market data
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            self.logger.info("🎓 Training HMM-based Market Regime Classifier...")

            # Calculate features with enhanced logging
            features_df = self._calculate_features(historical_klines)
            if features_df.empty:
                self.logger.error("No features available for HMM training")
                return False

            # Prepare features for HMM with proper scaling
            hmm_features = features_df[
                [
                    "log_returns",
                    "volatility_20",
                    "volume_ratio",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_histogram",
                    "bb_position",
                    "bb_width",
                    "atr",
                    "adx",
                    "volatility_regime",
                    "volatility_acceleration",
                ]
            ].fillna(0)

            # Enhanced feature validation
            self.logger.info(
                {
                    "msg": "HMM training features prepared",
                    "feature_shape": hmm_features.shape,
                    "feature_columns": list(hmm_features.columns),
                    "feature_stats": {
                        "mean": hmm_features.mean().to_dict(),
                        "std": hmm_features.std().to_dict(),
                        "min": hmm_features.min().to_dict(),
                        "max": hmm_features.max().to_dict(),
                    },
                }
            )

            # Scale features with proper error handling
            self.scaler = StandardScaler()
            hmm_features_scaled = self.scaler.fit_transform(hmm_features)

            # Train HMM model with enhanced configuration
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_states,
                n_iter=self.n_iter,
                random_state=self.random_state,
                covariance_type="full",
            )

            self.logger.info(f"🔧 Training HMM with {self.n_states} states, {self.n_iter} iterations...")
            self.hmm_model.fit(hmm_features_scaled)

            # Get state sequence and validate
            state_sequence = self.hmm_model.predict(hmm_features_scaled)
            
            # Log HMM training results
            unique_states = np.unique(state_sequence)
            state_counts = {int(state): int((state_sequence == state).sum()) for state in unique_states}
            
            self.logger.info(
                {
                    "msg": "HMM training completed",
                    "n_states": self.n_states,
                    "unique_states_found": len(unique_states),
                    "state_counts": state_counts,
                    "convergence_score": self.hmm_model.score(hmm_features_scaled),
                }
            )

            # Interpret states and create regime mapping
            state_analysis = self._interpret_hmm_states(features_df, state_sequence)
            self.state_to_regime_map = state_analysis["state_to_regime_map"]

            self.logger.info("✅ HMM-based regime classifier trained successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to train HMM regime classifier: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="UnifiedRegimeClassifier.train_location_classifier",
    )
    async def train_location_classifier(self, historical_klines: pd.DataFrame) -> bool:
        """
        Train location classifier for OPEN_RANGE, PIVOT_S, PIVOT_R, HVN_SUPPORT, HVN_RESISTANCE, CONFLUENCE_S, CONFLUENCE_R.
        
        Args:
            historical_klines: Historical market data
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            self.logger.info("🎓 Training Location Classifier...")

            # Calculate features
            features_df = self._calculate_features(historical_klines)
            if features_df.empty:
                self.logger.error("No features available for location classifier training")
                return False

            # Check if we have enough data for location classification
            long_term_hvn_period = self.config.get("long_term_hvn_period", 720)
            if len(features_df) < long_term_hvn_period:
                self.logger.warning(
                    f"Insufficient data for location classification. "
                    f"Need at least {long_term_hvn_period} rows, but only have {len(features_df)}. "
                    f"Skipping location classifier training."
                )
                return True  # Return True to avoid breaking the pipeline

            # Get location labels using the new _classify_location method
            location_labels = self._classify_location(features_df)

            # Verify that location labels match the features length
            if len(location_labels) != len(features_df):
                self.logger.error(
                    f"Location labels length ({len(location_labels)}) does not match "
                    f"features length ({len(features_df)}). Skipping location classifier training."
                )
                return True  # Return True to avoid breaking the pipeline

            # Encode location labels
            self.location_label_encoder = LabelEncoder()
            location_encoded = self.location_label_encoder.fit_transform(location_labels)

            # Prepare features for location classification
            location_features = features_df[
                ["close", "volume", "volatility_20", "rsi", "bb_position", "atr"]
            ].fillna(0)

            # Train location classifier with enhanced configuration
            self.location_classifier = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1,
            )

            self.location_classifier.fit(location_features, location_encoded)

            # Log location classifier training results
            unique_locations = np.unique(location_labels)
            location_counts = {loc: int(location_labels.count(loc)) for loc in unique_locations}
            
            self.logger.info(
                {
                    "msg": "Location classifier training completed",
                    "unique_locations": len(unique_locations),
                    "location_counts": location_counts,
                    "feature_shape": location_features.shape,
                }
            )

            self.logger.info("✅ Location classifier trained successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to train location classifier: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="UnifiedRegimeClassifier.train_basic_ensemble",
    )
    async def train_basic_ensemble(self, historical_klines: pd.DataFrame) -> bool:
        """
        Train ensemble for basic regime classification (BULL, BEAR, SIDEWAYS).
        
        Args:
            historical_klines: Historical market data
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            self.logger.info("🎓 Training Basic Regime Ensemble...")

            # Calculate features
            features_df = self._calculate_features(historical_klines)
            if features_df.empty:
                self.logger.error("No features available for ensemble training")
                return False

            # Get HMM-based labels using the balanced regime mapping
            hmm_features = features_df[
                [
                    "log_returns",
                    "volatility_20",
                    "volume_ratio",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_histogram",
                    "bb_position",
                    "bb_width",
                    "atr",
                    "adx",
                    "volatility_regime",
                    "volatility_acceleration",
                ]
            ].fillna(0)
            
            if self.scaler is None:
                self.logger.error("Scaler not available for ensemble training")
                return False
                
            hmm_features_scaled = self.scaler.transform(hmm_features)
            state_sequence = self.hmm_model.predict(hmm_features_scaled)

            # Map states to regimes using the balanced regime mapping
            regime_labels = [
                self.state_to_regime_map.get(state, "SIDEWAYS")
                for state in state_sequence
            ]
            
            # Log the regime distribution from the balanced mapping
            regime_counts = {}
            for regime in regime_labels:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            total_labels = len(regime_labels)
            regime_distribution = {regime: count/total_labels for regime, count in regime_counts.items()}
            
            self.logger.info(
                {
                    "msg": "Ensemble training regime distribution (balanced)",
                    "regime_distribution": regime_distribution,
                    "total_labels": total_labels,
                }
            )
            
            # Verify that the balanced regime mapping is being used correctly
            sideways_ratio = regime_counts.get("SIDEWAYS", 0) / total_labels if total_labels > 0 else 0
            if sideways_ratio > 0.40:
                self.logger.warning(f"⚠️ SIDEWAYS ratio in ensemble training data is {sideways_ratio:.3f} > 0.40 - balancing may not be working correctly")
            else:
                self.logger.info(f"✅ SIDEWAYS ratio in ensemble training data is {sideways_ratio:.3f} (within acceptable range)")

            # Encode regime labels
            self.basic_label_encoder = LabelEncoder()
            regime_encoded = self.basic_label_encoder.fit_transform(regime_labels)

            # Prepare features for ensemble
            ensemble_features = features_df[
                [
                    "log_returns",
                    "volatility_20",
                    "volume_ratio",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_histogram",
                    "bb_position",
                    "bb_width",
                    "atr",
                    "adx",
                    "volatility_regime",
                    "volatility_acceleration",
                ]
            ].fillna(0)

            # Train ensemble with class balancing
            self.basic_ensemble = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1,
                class_weight='balanced',  # Add class balancing
                is_unbalance=True,  # LightGBM specific parameter for imbalanced data
            )

            self.basic_ensemble.fit(ensemble_features, regime_encoded)

            # Log training results
            train_predictions = self.basic_ensemble.predict(ensemble_features)
            train_regime_labels = self.basic_label_encoder.inverse_transform(train_predictions)
            
            train_counts = {}
            for regime in train_regime_labels:
                train_counts[regime] = train_counts.get(regime, 0) + 1
            
            train_total = len(train_regime_labels)
            train_distribution = {regime: count/train_total for regime, count in train_counts.items()}
            
            self.logger.info(
                {
                    "msg": "Ensemble training results",
                    "train_distribution": train_distribution,
                    "train_total": train_total,
                }
            )
            
            # Check if training maintained the balance
            train_sideways_ratio = train_counts.get("SIDEWAYS", 0) / train_total if train_total > 0 else 0
            if train_sideways_ratio > 0.40:
                self.logger.warning(f"⚠️ Ensemble training produced SIDEWAYS ratio {train_sideways_ratio:.3f} > 0.40")
            else:
                self.logger.info(f"✅ Ensemble training maintained SIDEWAYS ratio {train_sideways_ratio:.3f} (within acceptable range)")

            self.logger.info("✅ Basic regime ensemble trained successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to train basic ensemble: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="UnifiedRegimeClassifier.train_complete_system",
    )
    async def train_complete_system(self, historical_klines: pd.DataFrame) -> bool:
        """
        Train the complete regime and location classification system.
        
        Args:
            historical_klines: Historical market data
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            self.logger.info("🎓 Training Complete Regime Classification System...")

            # Train HMM labeler
            if not await self.train_hmm_labeler(historical_klines):
                self.logger.error("❌ HMM labeler training failed")
                return False

            # Train basic ensemble
            if not await self.train_basic_ensemble(historical_klines):
                self.logger.error("❌ Basic ensemble training failed")
                return False

            # Train location classifier
            if not await self.train_location_classifier(historical_klines):
                self.logger.error("❌ Location classifier training failed")
                return False

            self.trained = True
            self.last_training_time = datetime.now()

            self.logger.info("✅ Complete regime classification system trained successfully")
            
            # Persist trained models so subsequent runs can load them
            self.save_models()
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to train complete system: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=("SIDEWAYS", 0.5, {"error": "Prediction failed"}),
        context="UnifiedRegimeClassifier.predict_regime",
    )
    def predict_regime(
        self,
        current_klines: pd.DataFrame,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Predict only the regime (for backward compatibility).

        Args:
            current_klines: Current market data

        Returns:
            Tuple of (regime, confidence, additional_info)
        """
        try:
            if not self.trained:
                self.logger.warning("Models not trained, returning default prediction")
                return "SIDEWAYS", 0.5, {}

            # Calculate features
            features_df = self._calculate_features(current_klines)
            if features_df.empty:
                self.logger.warning("No features available for prediction")
                return "SIDEWAYS", 0.5, {}

            current_features = features_df.iloc[-1] if len(features_df) > 0 else None
            if current_features is None:
                self.logger.warning("No current features available for prediction")
                return "SIDEWAYS", 0.5, {}

            # Predict regime
            regime_features = features_df[
                [
                    "log_returns",
                    "volatility_20",
                    "volume_ratio",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_histogram",
                    "bb_position",
                    "bb_width",
                    "atr",
                    "volatility_regime",
                    "volatility_acceleration",
                ]
            ].fillna(0)

            if self.basic_ensemble and self.basic_label_encoder:
                regime_proba = self.basic_ensemble.predict_proba(regime_features.iloc[-1:])
                regime_pred = self.basic_ensemble.predict(regime_features.iloc[-1:])[0]
                regime = self.basic_label_encoder.inverse_transform([regime_pred])[0]
                regime_confidence = float(np.max(regime_proba))
                
                self.logger.info(
                    {
                        "msg": "Regime prediction completed",
                        "predicted_regime": regime,
                        "confidence": regime_confidence,
                        "probabilities": regime_proba[0].tolist(),
                    }
                )
            else:
                regime = "SIDEWAYS"
                regime_confidence = 0.5
                self.logger.warning("Ensemble model not available, using default prediction")

            additional_info = {
                "regime_confidence": regime_confidence,
                "features_used": list(features_df.columns),
                "prediction_time": datetime.now().isoformat(),
            }

            return regime, regime_confidence, additional_info

        except Exception as e:
            self.logger.error(f"❌ Error in regime prediction: {e}")
            return "SIDEWAYS", 0.5, {"error": str(e)}

    @handle_errors(
        exceptions=(Exception,),
        default_return=("SIDEWAYS", "OPEN_RANGE", 0.5, {"error": "Prediction failed"}),
        context="UnifiedRegimeClassifier.predict_regime_and_location",
    )
    def predict_regime_and_location(
        self,
        current_klines: pd.DataFrame,
    ) -> Tuple[str, str, float, Dict[str, Any]]:
        """
        Predict both regime and location.

        Args:
            current_klines: Current market data

        Returns:
            Tuple of (regime, location, confidence, additional_info)
        """
        try:
            if not self.trained:
                self.logger.warning("Models not trained, returning default predictions")
                return "SIDEWAYS", "OPEN_RANGE", 0.5, {}

            # Calculate features
            features_df = self._calculate_features(current_klines)
            if features_df.empty:
                self.logger.warning("No features available for prediction")
                return "SIDEWAYS", "OPEN_RANGE", 0.5, {}

            current_features = features_df.iloc[-1] if len(features_df) > 0 else None
            if current_features is None:
                self.logger.warning("No current features available for prediction")
                return "SIDEWAYS", "OPEN_RANGE", 0.5, {}

            # Predict regime
            regime_features = features_df[
                [
                    "log_returns",
                    "volatility_20",
                    "volume_ratio",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_histogram",
                    "bb_position",
                    "bb_width",
                    "atr",
                    "volatility_regime",
                    "volatility_acceleration",
                ]
            ].fillna(0)

            if self.basic_ensemble and self.basic_label_encoder:
                regime_proba = self.basic_ensemble.predict_proba(regime_features.iloc[-1:])
                regime_pred = self.basic_ensemble.predict(regime_features.iloc[-1:])[0]
                regime = self.basic_label_encoder.inverse_transform([regime_pred])[0]
                regime_confidence = float(np.max(regime_proba))
            else:
                regime = "SIDEWAYS"
                regime_confidence = 0.5

            # Predict location
            location_features = features_df[
                ["close", "volume", "volatility_20", "rsi", "bb_position", "atr"]
            ].fillna(0)

            if self.location_classifier and self.location_label_encoder:
                location_proba = self.location_classifier.predict_proba(location_features.iloc[-1:])
                location_pred = self.location_classifier.predict(location_features.iloc[-1:])[0]
                location = self.location_label_encoder.inverse_transform([location_pred])[0]
                location_confidence = float(np.max(location_proba))
            else:
                # Fallback to rule-based location classification
                location_labels = self._classify_location(features_df)
                location = location_labels[-1] if location_labels else "OPEN_RANGE"
                location_confidence = 0.7

            # Calculate overall confidence
            overall_confidence = (regime_confidence + location_confidence) / 2

            self.logger.info(
                {
                    "msg": "Regime and location prediction completed",
                    "predicted_regime": regime,
                    "predicted_location": location,
                    "regime_confidence": regime_confidence,
                    "location_confidence": location_confidence,
                    "overall_confidence": overall_confidence,
                }
            )

            additional_info = {
                "regime_confidence": regime_confidence,
                "location_confidence": location_confidence,
                "features_used": list(features_df.columns),
                "prediction_time": datetime.now().isoformat(),
            }

            return regime, location, overall_confidence, additional_info

        except Exception as e:
            self.logger.error(f"❌ Error in regime/location prediction: {e}")
            return "SIDEWAYS", "OPEN_RANGE", 0.5, {"error": str(e)}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="UnifiedRegimeClassifier.save_models",
    )
    def save_models(self) -> None:
        """Save all trained models with enhanced error handling and logging."""
        try:
            self.logger.info("💾 Saving trained models...")
            
            if self.hmm_model:
                joblib.dump(self.hmm_model, self.hmm_model_path)
                self.logger.info(f"✅ HMM model saved to {self.hmm_model_path}")

            if self.basic_ensemble:
                joblib.dump(self.basic_ensemble, self.ensemble_model_path)
                self.logger.info(f"✅ Basic ensemble saved to {self.ensemble_model_path}")

            if self.location_classifier:
                joblib.dump(self.location_classifier, self.location_model_path)
                self.logger.info(f"✅ Location classifier saved to {self.location_model_path}")

            # Save label encoders
            if self.basic_label_encoder:
                encoder_path = self.ensemble_model_path.replace(".joblib", "_encoder.joblib")
                joblib.dump(self.basic_label_encoder, encoder_path)
                self.logger.info(f"✅ Basic label encoder saved to {encoder_path}")

            if self.location_label_encoder:
                encoder_path = self.location_model_path.replace(".joblib", "_encoder.joblib")
                joblib.dump(self.location_label_encoder, encoder_path)
                self.logger.info(f"✅ Location label encoder saved to {encoder_path}")

            self.logger.info("✅ All models saved successfully")

        except Exception as e:
            self.logger.error(f"❌ Error saving models: {e}")
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="UnifiedRegimeClassifier.load_models",
    )
    def load_models(self) -> bool:
        """Load all trained models with enhanced error handling and logging."""
        try:
            # Log model directory and candidate paths
            self.logger.info(
                {
                    "msg": "Loading models from directories",
                    "model_dir": self.model_dir,
                    "hmm_model_path": self.hmm_model_path,
                    "ensemble_model_path": self.ensemble_model_path,
                    "location_model_path": self.location_model_path,
                    "hierarchical_model_dir": getattr(self, "_hierarchical_model_dir", None),
                }
            )

            loaded_any = False

            # Candidate paths (flat first, then optional hierarchical forms)
            hmm_candidates = [
                self.hmm_model_path,
                os.path.join(
                    getattr(self, "_hierarchical_model_dir", self.model_dir),
                    "unified_hmm_model.joblib",
                ),
            ]
            ensemble_candidates = [
                self.ensemble_model_path,
                os.path.join(
                    getattr(self, "_hierarchical_model_dir", self.model_dir),
                    "unified_ensemble_model.joblib",
                ),
            ]
            location_candidates = [
                self.location_model_path,
                os.path.join(
                    getattr(self, "_hierarchical_model_dir", self.model_dir),
                    "unified_location_model.joblib",
                ),
            ]

            def _first_existing(paths: List[str]) -> Optional[str]:
                for p in paths:
                    if os.path.exists(p):
                        return p
                return None

            # Load HMM model
            hmm_path = _first_existing(hmm_candidates)
            if hmm_path is not None:
                self.hmm_model = joblib.load(hmm_path)
                self.logger.info(f"✅ Loaded HMM model from {hmm_path}")
                loaded_any = True

            # Load basic ensemble
            ensemble_path = _first_existing(ensemble_candidates)
            if ensemble_path is not None:
                self.basic_ensemble = joblib.load(ensemble_path)
                self.logger.info(f"✅ Loaded basic ensemble from {ensemble_path}")
                loaded_any = True

            # Load location classifier
            location_path = _first_existing(location_candidates)
            if location_path is not None:
                self.location_classifier = joblib.load(location_path)
                self.logger.info(f"✅ Loaded location classifier from {location_path}")
                loaded_any = True

            # Load label encoders
            encoder_candidates = [
                self.ensemble_model_path.replace(".joblib", "_encoder.joblib"),
                os.path.join(
                    getattr(self, "_hierarchical_model_dir", self.model_dir),
                    "unified_ensemble_model_encoder.joblib",
                ),
            ]
            enc_path = _first_existing(encoder_candidates)
            if enc_path is not None:
                self.basic_label_encoder = joblib.load(enc_path)
                self.logger.info(f"✅ Loaded basic label encoder from {enc_path}")

            location_encoder_candidates = [
                self.location_model_path.replace(".joblib", "_encoder.joblib"),
                os.path.join(
                    getattr(self, "_hierarchical_model_dir", self.model_dir),
                    "unified_location_model_encoder.joblib",
                ),
            ]
            loc_enc_path = _first_existing(location_encoder_candidates)
            if loc_enc_path is not None:
                self.location_label_encoder = joblib.load(loc_enc_path)
                self.logger.info(f"✅ Loaded location label encoder from {loc_enc_path}")

            self.trained = loaded_any
            
            if loaded_any:
                self.logger.info("✅ Model loading completed successfully")
            else:
                self.logger.info("ℹ️ No models found to load")
                
            return loaded_any

        except Exception as e:
            self.logger.error(f"❌ Error loading models: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={"error": "Classification failed"},
        context="UnifiedRegimeClassifier.classify_regimes",
    )
    async def classify_regimes(self, historical_klines: pd.DataFrame) -> Dict[str, Any]:
        """
        Classify regimes for historical data (for training purposes).

        Args:
            historical_klines: Historical market data

        Returns:
            Dict containing regime classification results
        """
        try:
            if not self.trained:
                self.logger.info("🎓 Models not trained, training complete system now...")
                training_success = await self.train_complete_system(historical_klines)
                if not training_success:
                    self.logger.error("❌ Failed to train regime classification models")
                    return {"error": "Failed to train regime classification models"}

            # Calculate features
            features_df = self._calculate_features(historical_klines)
            if features_df.empty:
                self.logger.error("❌ No features available for classification")
                return {"error": "No features available for classification"}

            # Get regime predictions
            regime_features = features_df[
                [
                    "log_returns",
                    "volatility_20",
                    "volume_ratio",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_histogram",
                    "bb_position",
                    "bb_width",
                    "atr",
                    "adx",
                    "volatility_regime",
                    "volatility_acceleration",
                ]
            ].fillna(0)

            regimes = []
            confidence_scores = []
            
            if self.basic_ensemble and self.basic_label_encoder:
                self.logger.info("🔍 Using trained basic ensemble for regime classification")
                regime_predictions = self.basic_ensemble.predict(regime_features)
                regime_probabilities = self.basic_ensemble.predict_proba(regime_features)
                regimes = self.basic_label_encoder.inverse_transform(regime_predictions).tolist()
                # Calculate confidence scores as max probability for each prediction
                confidence_scores = [float(np.max(proba)) for proba in regime_probabilities]
                
                unique_regimes = list(sorted(set(regimes)))
                counts = {r: int((np.array(regimes) == r).sum()) for r in unique_regimes}
                
                # Detailed logging on regime prediction composition
                self.logger.info(
                    {
                        "msg": "Ensemble regime prediction summary",
                        "unique_regimes": unique_regimes,
                        "counts": counts,
                        "expected_min": self.n_states,
                        "total_records": int(len(regime_features)),
                        "thresholds": {
                            "adx_sideways_threshold": self.adx_sideways_threshold,
                            "volatility_threshold": self.volatility_threshold,
                            "atr_normalized_threshold": self.atr_normalized_threshold,
                            "volatility_percentile_threshold": self.volatility_percentile_threshold,
                        },
                    }
                )
                
                # Compare ensemble predictions with HMM predictions for validation
                if self.hmm_model and self.scaler and self.state_to_regime_map:
                    hmm_features_scaled = self.scaler.transform(regime_features)
                    hmm_state_sequence = self.hmm_model.predict(hmm_features_scaled)
                    hmm_regimes = [
                        self.state_to_regime_map.get(state, "SIDEWAYS")
                        for state in hmm_state_sequence
                    ]
                    
                    hmm_counts = {}
                    for regime in hmm_regimes:
                        hmm_counts[regime] = hmm_counts.get(regime, 0) + 1
                    
                    ensemble_counts = {}
                    for regime in regimes:
                        ensemble_counts[regime] = ensemble_counts.get(regime, 0) + 1
                    
                    self.logger.info(f"📊 Model comparison - HMM: {hmm_counts}, Ensemble: {ensemble_counts}")
                    
                    # Calculate agreement rate
                    agreement = sum(1 for e, h in zip(regimes, hmm_regimes) if e == h) / len(regimes)
                    self.logger.info(f"📊 Model agreement rate: {agreement:.3f}")
                
                if len(unique_regimes) < self.n_states:
                    self.logger.warning(
                        warning(
                            f"Fewer regimes predicted ({len(unique_regimes)}) than expected ({self.n_states}). "
                            "Consider increasing min_data_points or enhancing volatility features."
                        )
                    )
                    
            # Fallback to HMM states
            elif self.hmm_model and self.scaler and self.state_to_regime_map:
                self.logger.info("🔍 Using HMM model for regime classification")
                hmm_features_scaled = self.scaler.transform(regime_features)
                state_sequence = self.hmm_model.predict(hmm_features_scaled)
                regimes = [
                    self.state_to_regime_map.get(state, "SIDEWAYS")
                    for state in state_sequence
                ]
                # For HMM, use a default confidence score since we don't have probabilities
                confidence_scores = [0.8] * len(regimes)  # Default high confidence for HMM
                unique_regimes = list(sorted(set(regimes)))
                counts = {r: int((np.array(regimes) == r).sum()) for r in unique_regimes}
                
                self.logger.info(
                    {
                        "msg": "HMM regime prediction summary",
                        "unique_regimes": unique_regimes,
                        "counts": counts,
                        "expected_min": self.n_states,
                        "total_records": int(len(regime_features)),
                    }
                )
                
                if len(unique_regimes) < self.n_states:
                    self.logger.warning(
                        warning(
                            f"Fewer regimes predicted ({len(unique_regimes)}) than expected ({self.n_states}). "
                            "Consider increasing min_data_points or enhancing volatility features."
                        )
                    )
            else:
                self.logger.warning("⚠️ No trained models available, attempting to train models now...")
                # Try to train the complete system
                training_success = await self.train_complete_system(historical_klines)
                if not training_success:
                    self.logger.error("❌ Failed to train regime classification models")
                    return {"error": "Failed to train regime classification models"}

                # Retry classification after training
                self.logger.info("🔄 Retrying regime classification with newly trained models...")
                return await self.classify_regimes(historical_klines)

            # Get location predictions
            location_labels = self._classify_location(features_df)

            regime_distribution = dict(pd.Series(regimes).value_counts())
            # Convert numpy types to regular Python types for clean logging
            clean_distribution = {k: int(v) for k, v in regime_distribution.items()}
            self.logger.info(f"📊 Regime distribution: {clean_distribution}")

            return {
                "regimes": regimes,
                "confidence_scores": confidence_scores,
                "locations": location_labels,
                "total_records": len(features_df),
                "regime_distribution": regime_distribution,
                "location_distribution": dict(pd.Series(location_labels).value_counts()),
            }

        except Exception as e:
            self.logger.exception(f"❌ Error in regime classification: {e}")
            return {"error": str(e)}

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics with enhanced information."""
        return {
            "trained": self.trained,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "hmm_model_loaded": self.hmm_model is not None,
            "basic_ensemble_loaded": self.basic_ensemble is not None,
            "location_classifier_loaded": self.location_classifier is not None,
            "n_states": self.n_states,
            "target_timeframe": self.target_timeframe,
            "state_to_regime_map": self.state_to_regime_map,
            "model_paths": {
                "model_dir": self.model_dir,
                "hmm_model_path": self.hmm_model_path,
                "ensemble_model_path": self.ensemble_model_path,
                "location_model_path": self.location_model_path,
            },
            "configuration": {
                "adx_sideways_threshold": self.adx_sideways_threshold,
                "volatility_threshold": self.volatility_threshold,
                "atr_normalized_threshold": self.atr_normalized_threshold,
                "volatility_percentile_threshold": self.volatility_percentile_threshold,
                "bb_width_volatility_threshold": self.bb_width_volatility_threshold,
            },
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="UnifiedRegimeClassifier._calculate_rolling_pivots",
    )
    def _calculate_rolling_pivots(self, df_window: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate rolling pivot points for dynamic support and resistance with strength metrics.

        Args:
            df_window: DataFrame window for pivot calculation

        Returns:
            Dict containing pivot levels with strength metrics
        """
        try:
            if len(df_window) < 5:
                return {
                    "s1": 0,
                    "s2": 0,
                    "r1": 0,
                    "r2": 0,
                    "pivot": 0,
                    "strengths": {
                        "s1": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0},
                        "s2": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0},
                        "r1": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0},
                        "r2": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0},
                    },
                }

            # Calculate pivot point
            high = df_window["high"].max()
            low = df_window["low"].min()
            close = df_window["close"].iloc[-1]

            pivot = (high + low + close) / 3

            # Calculate support and resistance levels
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)

            # Calculate strength metrics for each level
            levels = {"s1": s1, "s2": s2, "r1": r1, "r2": r2}
            strengths = {}

            for level_name, level_price in levels.items():
                if level_price <= 0:
                    strengths[level_name] = {
                        "strength": 0.0,
                        "touches": 0,
                        "volume": 0.0,
                        "age": 0,
                    }
                    continue

                # Calculate touches (how many times price approached this level)
                touches = 0
                tolerance = df_window["close"].std() * 0.1  # 10% of price volatility

                for i in range(1, len(df_window)):
                    curr_close = df_window["close"].iloc[i]

                    # Check if price approached the level
                    if abs(curr_close - level_price) <= tolerance:
                        touches += 1

                # Calculate volume near this level
                volume_near_level = 0.0
                for i in range(len(df_window)):
                    if abs(df_window["close"].iloc[i] - level_price) <= tolerance:
                        volume_near_level += df_window["volume"].iloc[i]

                # Calculate age (how long ago this level was first established)
                age = 0
                for i in range(len(df_window)):
                    if abs(df_window["close"].iloc[i] - level_price) <= tolerance:
                        age = len(df_window) - i
                        break

                # Calculate overall strength (0.0 to 1.0)
                # Factors: touches (30%), volume (40%), age (30%)
                touch_strength = min(touches / 5.0, 1.0)  # Normalize touches
                volume_strength = min(
                    volume_near_level / df_window["volume"].sum(),
                    1.0,
                )  # Normalize volume
                age_strength = min(age / len(df_window), 1.0)  # Normalize age

                overall_strength = (
                    touch_strength * 0.3 + volume_strength * 0.4 + age_strength * 0.3
                )

                strengths[level_name] = {
                    "strength": overall_strength,
                    "touches": touches,
                    "volume": volume_near_level,
                    "age": age,
                }

            return {
                "s1": s1,
                "s2": s2,
                "r1": r1,
                "r2": r2,
                "pivot": pivot,
                "strengths": strengths,
            }

        except Exception as e:
            self.logger.error(f"Error calculating rolling pivots: {e}")
            return {
                "s1": 0,
                "s2": 0,
                "r1": 0,
                "r2": 0,
                "pivot": 0,
                "strengths": {
                    "s1": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0},
                    "s2": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0},
                    "r1": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0},
                    "r2": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0},
                },
            }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="UnifiedRegimeClassifier._analyze_volume_levels",
    )
    def _analyze_volume_levels(self, df_window: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyzes the volume profile to find the two most significant High Volume Nodes (HVNs),
        their age, and the number of times they've been tested.
        
        OPTIMIZED VERSION: Reduced complexity for better performance
        
        Args:
            df_window: DataFrame window for volume analysis
            
        Returns:
            Dict containing volume level analysis or None if insufficient data
        """
        try:
            if df_window.empty or len(df_window) < 20:
                return None

            # --- 1. ATR-Dynamic Binning (same as before) ---
            high_low = df_window["high"] - df_window["low"]
            high_close = abs(df_window["high"] - df_window["close"].shift())
            low_close = abs(df_window["low"] - df_window["close"].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            avg_atr = tr.mean()
            bin_size = max(avg_atr * 0.25, 1e-6)
            min_price = df_window["low"].min()
            max_price = df_window["high"].max()
            bins = np.arange(min_price, max_price, bin_size)

            # --- 2. Find Top 2 HVNs ---
            price_bins = pd.cut(df_window["close"], bins=bins, right=False)
            volume_by_bin = df_window.groupby(price_bins, observed=False)["volume"].sum()
            if volume_by_bin.empty:
                return None

            # Get the top 2 HVNs by volume
            top_hvns = volume_by_bin.nlargest(2)
            if top_hvns.empty:
                return None

            # --- 3. Analyze Each HVN (OPTIMIZED) ---
            analyzed_levels = {}
            for i, (level_bin, level_volume) in enumerate(top_hvns.items()):
                level_price = level_bin.mid

                # Find first time price entered this bin to determine age
                level_indices = df_window.index[
                    df_window["close"].between(level_bin.left, level_bin.right)
                ]
                if len(level_indices) == 0:
                    continue

                first_touch_index = level_indices[0]
                age = len(df_window) - df_window.index.get_loc(first_touch_index)

                # OPTIMIZATION: Simplified touch counting using vectorized operations
                # Count touches more efficiently by checking price crosses
                touches = 0
                if len(df_window) > 1:
                    # Use vectorized operations to find price crosses
                    high_crosses = (df_window["high"] >= level_price) & (df_window["high"].shift() < level_price)
                    low_crosses = (df_window["low"] <= level_price) & (df_window["low"].shift() > level_price)
                    touches = (high_crosses | low_crosses).sum()

                # Calculate additional strength metrics
                # Volume strength (normalized)
                total_volume = df_window["volume"].sum()
                volume_strength = (
                    min(level_volume / total_volume, 1.0) if total_volume > 0 else 0.0
                )

                # Touch strength (normalized)
                touch_strength = min(touches / 10.0, 1.0)  # Normalize touches

                # Age strength (normalized)
                age_strength = min(age / len(df_window), 1.0)  # Normalize age

                # Calculate overall strength (0.0 to 1.0)
                # Factors: volume (50%), touches (30%), age (20%)
                overall_strength = (
                    volume_strength * 0.5 + touch_strength * 0.3 + age_strength * 0.2
                )

                level_name = "poc" if i == 0 else "hvn_secondary"
                analyzed_levels[level_name] = {
                    "price": level_price,
                    "volume": level_volume,
                    "age": age,  # in number of candles
                    "touches": touches,
                    "strength": overall_strength,
                    "volume_strength": volume_strength,
                    "touch_strength": touch_strength,
                    "age_strength": age_strength,
                }

            return analyzed_levels

        except Exception as e:
            self.logger.error(f"Error analyzing volume levels: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=["OPEN_RANGE"] * 1000,  # Safe fallback
        context="UnifiedRegimeClassifier._classify_location",
    )
    def _classify_location(self, features_df: pd.DataFrame) -> List[str]:
        """
        Classifies location using a multi-layered context of short-term Dynamic Pivots (tactical)
        and long-term High Volume Nodes (strategic).
        
        OPTIMIZED VERSION: Uses vectorized operations instead of per-point calculations
        
        Args:
            features_df: DataFrame with calculated features
            
        Returns:
            List of location classifications
        """
        try:
            self.logger.info("Classifying location with tactical pivots and strategic volume levels...")

            # --- Configuration for dual-timeframe analysis ---
            long_term_hvn_period = self.config.get("long_term_hvn_period", 720)  # 30 days on a 1h chart
            short_term_pivot_period = self.config.get("short_term_pivot_period", 24)  # 1 day on a 1h chart
            tolerance = self.config.get("level_tolerance", 0.01)  # 1% proximity tolerance
            min_level_touches = self.config.get("min_level_touches", 1)  # Must have at least 1 re-test

            # Check if we have enough data for location classification
            if len(features_df) < long_term_hvn_period:
                self.logger.warning(
                    f"Insufficient data for location classification. "
                    f"Need at least {long_term_hvn_period} rows, but only have {len(features_df)}. "
                    f"Returning all OPEN_RANGE labels."
                )
                return ["OPEN_RANGE"] * len(features_df)

            # Initialize all locations as OPEN_RANGE
            locations = ["OPEN_RANGE"] * len(features_df)
            
            # Calculate price differences once (vectorized)
            price_diffs = features_df["close"].diff()
            
            # Start processing after the longest period
            start_index = long_term_hvn_period
            
            # OPTIMIZATION: Calculate global volume levels once instead of per-point
            self.logger.info("🔍 Calculating global volume levels...")
            global_volume_levels = self._analyze_volume_levels(features_df)
            
            # Process in batches for better performance
            batch_size = 100
            total_batches = (len(features_df) - start_index) // batch_size + 1
            
            self.logger.info(f"📊 Processing {len(features_df) - start_index} records in {total_batches} batches...")
            
            for batch_start in range(start_index, len(features_df), batch_size):
                batch_end = min(batch_start + batch_size, len(features_df))
                batch_num = (batch_start - start_index) // batch_size + 1
                
                if batch_num % 10 == 0:  # Log progress every 10 batches
                    self.logger.info(f"🔄 Processing batch {batch_num}/{total_batches} (records {batch_start}-{batch_end})")
                
                for i in range(batch_start, batch_end):
                    current_price_diff = price_diffs.iloc[i]
                    
                    # Skip if price difference is NaN
                    if pd.isna(current_price_diff):
                        continue
                    
                    # --- 1. Tactical Pivot Analysis (Short-Term) ---
                    # Only calculate pivots if we have enough data
                    if i >= short_term_pivot_period:
                        pivot_window = features_df.iloc[i - short_term_pivot_period : i]
                        pivots = self._calculate_rolling_pivots(pivot_window)
                        pivot_supports = [pivots["s1"], pivots["s2"]]
                        pivot_resistances = [pivots["r1"], pivots["r2"]]
                    else:
                        pivot_supports = []
                        pivot_resistances = []

                    # --- 2. Strategic Volume Level Analysis (Long-Term) ---
                    # Use global volume levels instead of recalculating
                    volume_levels = global_volume_levels

                    # --- 3. Classification Logic ---
                    loc_pivot = None
                    loc_hvn = None

                    # Check for Pivot proximity using price differences
                    for p_sup in pivot_supports:
                        if p_sup > 0 and abs(current_price_diff - p_sup) / (abs(current_price_diff) + 1e-8) <= tolerance:
                            loc_pivot = "PIVOT_S"
                            break
                    if not loc_pivot:
                        for p_res in pivot_resistances:
                            if p_res > 0 and abs(current_price_diff - p_res) / (abs(current_price_diff) + 1e-8) <= tolerance:
                                loc_pivot = "PIVOT_R"
                                break

                    # Check for HVN proximity using price differences
                    if volume_levels:
                        for level_data in volume_levels.values():
                            # Intelligence Rule: Filter out untested levels
                            if level_data["touches"] < min_level_touches:
                                continue

                            if abs(current_price_diff - level_data["price"]) / (abs(current_price_diff) + 1e-8) <= tolerance:
                                hvn_type = (
                                    "SUPPORT"
                                    if current_price_diff > level_data["price"]
                                    else "RESISTANCE"
                                )
                                loc_hvn = f"HVN_{hvn_type}"
                                break  # Stop at the first significant HVN found

                    # --- 4. Final Label Assignment ---
                    if loc_pivot and loc_hvn:
                        # A tactical pivot aligns with a strategic volume level - high confluence
                        if "S" in loc_pivot and "SUPPORT" in loc_hvn:
                            locations[i] = "CONFLUENCE_S"
                        elif "R" in loc_pivot and "RESISTANCE" in loc_hvn:
                            locations[i] = "CONFLUENCE_R"
                        else:
                            locations[i] = loc_pivot
                    elif loc_pivot:
                        locations[i] = loc_pivot
                    elif loc_hvn:
                        locations[i] = loc_hvn
                    # else: already set to "OPEN_RANGE"

            self.logger.info(
                f"✅ Finished classifying locations. Found: {pd.Series(locations).value_counts().to_dict()}",
            )
            return locations

        except Exception as e:
            self.logger.error(f"Error in location classification: {e}")
            # Return safe fallback
            return ["OPEN_RANGE"] * len(features_df) if len(features_df) > 0 else ["OPEN_RANGE"]
