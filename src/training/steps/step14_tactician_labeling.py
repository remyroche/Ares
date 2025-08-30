# src/training/steps/step8_tactician_labeling.py

import asyncio
import contextlib
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.training.data_sharing_manager import get_data_sharing_manager
from src.training.steps.unified_data_loader import get_unified_data_loader
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger

# Preference order for selecting analyst ensembles
ENSEMBLE_PREFERENCE_ORDER = ("stacking_cv", "dynamic_weighting", "voting")

# Removing duplicate earlier TacticianLabelingStep definition to avoid conflicts


class TacticianTripleBarrierLabeler:
    """Applies triple barrier labeling for Tactician multi-outcome predictions.

    This labeler generates multi-outcome predictions similar to the Analyst but with:
    - Smaller price deviations (using Tactician's 50%/25% barriers)
    - Higher confidence for reaching target prices
    - Price direction predictions
    - Market regime detection
    - Volatility and momentum predictions
    
    Enhanced for high precision completion of Analyst signals with:
    - 50% smaller upper barriers (0.1% vs 0.2%)
    - 25% smaller lower barriers (0.025% vs 0.1%)
    - Higher confidence scores
    - Multi-outcome prediction structure
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config.get("tactician_triple_barrier", {})
        self.logger = system_logger.getChild("TacticianTripleBarrierLabeler")
        
        # Load enhanced configuration
        self._load_enhanced_config()

    def _load_enhanced_config(self) -> None:
        """Load enhanced configuration for high precision execution."""
        # Import dynamic barrier calculator
        from src.tactician.dynamic_barrier_calculator import DynamicBarrierCalculator
        
        # Initialize dynamic barrier calculator
        self.barrier_calculator = DynamicBarrierCalculator(self.config)
        
        # Get all 4 barrier combinations for primary timeframe (1m)
        self.barrier_combinations = self.barrier_calculator.calculate_dynamic_barriers(
            timeframe="1m"
        )
        
        # Get configuration for other settings
        self.max_lookahead = self.config.get("max_lookahead", 50)  # Reduced lookahead
        
        # Precision Settings
        self.enable_high_precision_mode = self.config.get("enable_high_precision_mode", True)
        self.precision_threshold = self.config.get("precision_threshold", 0.85)
        self.min_signal_strength = self.config.get("min_signal_strength", 0.8)
        
        # Quality Filters
        self.enable_quality_filters = self.config.get("enable_quality_filters", True)
        self.min_volume_threshold = self.config.get("min_volume_threshold", 1000)
        self.min_spread_threshold = self.config.get("min_spread_threshold", 0.0001)
        self.volatility_filter = self.config.get("volatility_filter", True)
        
        # Integration Settings
        self.analyst_signal_requirement = self.config.get("analyst_signal_requirement", True)
        self.direction_agreement_required = self.config.get("direction_agreement_required", True)
        self.confidence_boost_threshold = self.config.get("confidence_boost_threshold", 0.9)
        
        # Timeframe settings
        self.timeframes = self.config.get("timeframes", ["1m", "5m"])
        self.primary_timeframe = self.config.get("primary_timeframe", "1m")
        self.secondary_timeframe = self.config.get("secondary_timeframe", "5m")
        
        # Log configuration
        self.logger.info(f"🔧 Enhanced Tactician Triple Barrier Configuration (4 Barrier Combinations):")
        self.logger.info(f"   Timeframes: {self.timeframes}")
        self.logger.info(f"   Primary: {self.primary_timeframe}, Secondary: {self.secondary_timeframe}")
        self.logger.info(f"   4 Barrier Combinations:")
        for name, (upper, lower) in self.barrier_combinations.items():
            self.logger.info(f"     {name}: Upper={upper:.4f} ({upper*100:.3f}%), Lower={lower:.4f} ({lower*100:.3f}%)")
        self.logger.info(f"   High Precision Mode: {self.enable_high_precision_mode}")
        self.logger.info(f"   Precision Threshold: {self.precision_threshold}")

    def _apply_quality_filters(self, data: pd.DataFrame, entry_idx: int) -> bool:
        """Apply quality filters for high precision execution."""
        if not self.enable_quality_filters:
            return True
            
        try:
            # Volume filter
            if "volume" in data.columns:
                volume = data.iloc[entry_idx]["volume"]
                if volume < self.min_volume_threshold:
                    return False
            
            # Spread filter (if bid/ask data available)
            if "bid" in data.columns and "ask" in data.columns:
                bid = data.iloc[entry_idx]["bid"]
                ask = data.iloc[entry_idx]["ask"]
                spread = (ask - bid) / bid
                if spread > self.min_spread_threshold:
                    return False
            
            # Volatility filter
            if self.volatility_filter and len(data) >= 20:
                recent_data = data.iloc[max(0, entry_idx-20):entry_idx+1]
                if len(recent_data) >= 10:
                    returns = recent_data["close"].pct_change().dropna()
                    volatility = returns.std()
                    # Filter out extremely high volatility periods
                    if volatility > 0.01:  # 1% volatility threshold
                        return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Quality filter error: {e}")
            return True  # Default to allow if filter fails

    def _calculate_adaptive_barriers(self, data: pd.DataFrame, entry_idx: int, base_pt: float, base_sl: float) -> tuple[float, float]:
        """Calculate barriers - no dynamic adaptation, ML model handles market conditions."""
        # No hardcoded volatility adjustment - ML model will calculate optimal barriers
        return base_pt, base_sl

    def _generate_multi_outcome_predictions(
        self,
        data: pd.DataFrame,
        entry_idx: int,
        signal: int,
        barrier_combinations: Dict[str, Tuple[float, float]],
        entry_price: float,
        precision_score: float,
        quality_score: float
    ) -> dict[str, Any]:
        """Generate multi-outcome predictions for 2 barrier combinations."""
        try:
            # Calculate price deviations for 2 barrier combinations
            price_deviations = {}
            price_directions = {}
            price_target_confidences = {}
            
            for barrier_name, (upper_barrier, lower_barrier) in barrier_combinations.items():
                if signal == 1:  # Long position
                    # Calculate deviation to upper barrier
                    price_deviation = (upper_barrier - entry_price) / entry_price
                    price_direction = 1
                else:  # Short position
                    # Calculate deviation to lower barrier
                    price_deviation = (entry_price - lower_barrier) / entry_price
                    price_direction = -1
                
                # Store predictions for this barrier combination
                price_deviations[barrier_name] = price_deviation
                price_directions[barrier_name] = price_direction
                
                # Calculate confidence to reach upper barrier before lower barrier
                # ML model calculates this based on market conditions, volatility, and barrier distances
                base_confidence = precision_score
                # ML model will enhance confidence based on:
                # - Market volatility
                # - Barrier distances  
                # - Recent price action
                # - Support/resistance levels
                price_target_confidences[barrier_name] = base_confidence
            
            return {
                "price_deviations": price_deviations,
                "price_directions": price_directions,
                "price_target_confidences": price_target_confidences,
                "label": signal,  # Traditional label for backward compatibility
                "precision_score": precision_score,
                "execution_quality": quality_score
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating multi-outcome predictions: {e}")
            # Return fallback values for 2 barrier combinations
            fallback_deviations = {name: 0.0 for name in barrier_combinations.keys()}
            fallback_directions = {name: 0 for name in barrier_combinations.keys()}
            fallback_confidences = {name: 0.0 for name in barrier_combinations.keys()}
            
            return {
                "price_deviations": fallback_deviations,
                "price_directions": fallback_directions,
                "price_target_confidences": fallback_confidences,
                "label": signal,
                "precision_score": precision_score,
                "execution_quality": quality_score
            }

    def apply_labels(
        self, data: pd.DataFrame, strategic_signals: pd.Series,
    ) -> pd.DataFrame:
        """Vectorized application of the enhanced triple barrier method with multi-outcome predictions.

        Args:
            data: The 1-minute market data (must contain OHLC columns).
            strategic_signals: A Series with timestamps as index and signals (+1 for BUY, -1 for SELL)
                               as values, indicating when the Analyst has identified a setup.

        Returns:
            A DataFrame with multi-outcome predictions and precision metrics.
        """
        self.logger.info(
            "🔧 Applying Tactician multi-outcome predictions with enhanced triple barrier method...",
        )

        # Align signals with the data index
        entry_points = (
            strategic_signals[strategic_signals != 0].reindex(data.index).dropna()
        )
        if entry_points.empty:
            self.logger.warning(
                "⚠️ No strategic signals found to label. Returning data without labels.",
            )
            # Initialize multi-outcome predictions with defaults
            data["tactician_price_deviation"] = 0.0
            data["tactician_price_direction"] = 0
            data["tactician_price_target_confidence"] = 0.0
            data["tactician_regime"] = 0.0
            data["tactician_volatility"] = 0.0
            data["tactician_momentum"] = 0.0
            data["tactician_label"] = -1  # Default to sell signal for binary classification
            data["tactician_precision_score"] = 0.0
            return data

        entry_indices = data.index.get_indexer_for(entry_points.index)

        # Calculate fixed percentage barriers for each entry point
        entry_prices = data["open"].iloc[entry_indices + 1]

        # Initialize multi-outcome predictions (only 3 types)
        price_deviation_predictions = pd.Series(0.0, index=data.index)
        price_direction_predictions = pd.Series(0, index=data.index)
        price_target_confidence = pd.Series(0.0, index=data.index)
        
        # Initialize traditional labels for backward compatibility
        labels = pd.Series(-1, index=data.index)  # Default to sell signal
        precision_scores = pd.Series(0.0, index=data.index)
        execution_quality = pd.Series(0.0, index=data.index)

        # Vectorized barrier check with enhanced precision
        for i, entry_idx in enumerate(entry_indices):
            if entry_idx >= len(data) - 1:
                continue

            signal = entry_points.iloc[i]
            
            # Apply quality filters
            if not self._apply_quality_filters(data, entry_idx):
                continue
            
            # Calculate barriers for 2 combinations
            barrier_prices = {}
            for barrier_name, (upper_pct, lower_pct) in self.barrier_combinations.items():
                base_upper = entry_prices.iloc[i] * (1 + upper_pct * signal)
                base_lower = entry_prices.iloc[i] * (1 - lower_pct * signal)
                
                # No adaptive barriers - ML model handles market conditions
                upper, lower = base_upper, base_lower
                
                barrier_prices[barrier_name] = (upper, lower)

            # Get the path data for barrier hit detection
            path = data.iloc[entry_idx:entry_idx + self.max_lookahead]

            # Check for hits with enhanced precision for 2 barrier combinations
            barrier_results = {}
            best_precision_score = 0.0
            best_quality_score = 0.0
            best_barrier_name = None
            
            for barrier_name, (upper, lower) in barrier_prices.items():
                # Check for hits with enhanced precision (no time barrier)
                upper_hit_mask = (path["high"] >= upper) if signal == 1 else (path["low"] <= upper)
                lower_hit_mask = (path["low"] <= lower) if signal == 1 else (path["high"] >= lower)

                upper_hit_time = path.index[upper_hit_mask].min()
                lower_hit_time = path.index[lower_hit_mask].min()

                # Determine which barrier was hit first for this combination
                if pd.notna(upper_hit_time) and (
                    pd.isna(lower_hit_time) or upper_hit_time <= lower_hit_time
                ):
                    # Upper barrier hit first
                    time_to_hit = (upper_hit_time - data.index[entry_idx]).total_seconds() / 60  # minutes
                    precision_score = max(0.0, 1.0 - (time_to_hit / 30))  # Use 30 minutes as reference
                    # Get the barrier percentages for this combination
                    barrier_upper_pct, barrier_lower_pct = self.barrier_combinations[barrier_name]
                    quality_score = barrier_upper_pct / (barrier_upper_pct + barrier_lower_pct)  # Risk-reward ratio
                    barrier_result = 1  # Upper barrier hit
                    
                elif pd.notna(lower_hit_time):
                    # Lower barrier hit first
                    time_to_hit = (lower_hit_time - data.index[entry_idx]).total_seconds() / 60  # minutes
                    precision_score = max(0.0, 1.0 - (time_to_hit / 30)) * 0.5  # Penalty for lower barrier hit
                    quality_score = 0.0  # No quality score for lower barrier hit
                    barrier_result = -1  # Lower barrier hit
                    
                else:
                    # No barrier hit within lookahead
                    precision_score = 0.0
                    quality_score = 0.0
                    barrier_result = 0
                
                barrier_results[barrier_name] = {
                    "result": barrier_result,
                    "precision_score": precision_score,
                    "quality_score": quality_score,
                    "upper_hit_time": upper_hit_time,
                    "lower_hit_time": lower_hit_time
                }
                
                # Track the best performing barrier combination
                if precision_score > best_precision_score:
                    best_precision_score = precision_score
                    best_quality_score = quality_score
                    best_barrier_name = barrier_name

            # Use the best performing barrier combination for traditional labeling
            if best_barrier_name:
                best_result = barrier_results[best_barrier_name]
                labels.iloc[entry_idx] = best_result["result"]
                precision_score = best_result["precision_score"]
                quality_score = best_result["quality_score"]
            else:
                labels.iloc[entry_idx] = 0
                precision_score = 0.0
                quality_score = 0.0

            # Generate multi-outcome predictions for 2 barrier combinations
            predictions = self._generate_multi_outcome_predictions(
                data, entry_idx, signal, barrier_prices, entry_prices.iloc[i], precision_score, quality_score
            )
            
            # Store multi-outcome predictions for 2 barrier combinations
            # Store the best performing barrier combination as the main prediction
            if best_barrier_name:
                price_deviation_predictions.iloc[entry_idx] = predictions["price_deviations"][best_barrier_name]
                price_direction_predictions.iloc[entry_idx] = predictions["price_directions"][best_barrier_name]
                price_target_confidence.iloc[entry_idx] = predictions["price_target_confidences"][best_barrier_name]
            else:
                price_deviation_predictions.iloc[entry_idx] = 0.0
                price_direction_predictions.iloc[entry_idx] = 0
                price_target_confidence.iloc[entry_idx] = 0.0
            
            # Apply high precision mode filtering
            if self.enable_high_precision_mode:
                if precision_score < self.precision_threshold:
                    labels.iloc[entry_idx] = 0  # Filter out low precision signals
                    precision_score = 0.0
                    # Reset multi-outcome predictions for low precision signals
                    price_deviation_predictions.iloc[entry_idx] = 0.0
                    price_direction_predictions.iloc[entry_idx] = 0
                    price_target_confidence.iloc[entry_idx] = 0.0

            precision_scores.iloc[entry_idx] = precision_score
            execution_quality.iloc[entry_idx] = quality_score

        # Add multi-outcome predictions to data (only 3 types)
        data["tactician_price_deviation"] = price_deviation_predictions
        data["tactician_price_direction"] = price_direction_predictions
        data["tactician_price_target_confidence"] = price_target_confidence
        
        # Add traditional labels for backward compatibility
        data["tactician_label"] = labels
        data["tactician_precision_score"] = precision_scores
        data["tactician_execution_quality"] = execution_quality
        
        # Log results
        label_distribution = labels.value_counts()
        avg_precision = precision_scores[precision_scores > 0].mean()
        avg_quality = execution_quality[execution_quality > 0].mean()
        avg_confidence = price_target_confidence[price_target_confidence > 0].mean()
        
        self.logger.info(f"Tactician multi-outcome predictions complete:")
        self.logger.info(f"   Label distribution: {label_distribution}")
        self.logger.info(f"   Average precision score: {avg_precision:.3f}")
        self.logger.info(f"   Average execution quality: {avg_quality:.3f}")
        self.logger.info(f"   Average price target confidence: {avg_confidence:.3f}")
        self.logger.info(f"   High precision signals: {(precision_scores >= self.precision_threshold).sum()}")
        
        return data


class TacticianLabelingStep:
    """Step 8: Tactician Model Labeling using Analyst's model."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="tactician labeling step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the tactician labeling step."""
        self.logger.info("🚀 Initializing Tactician Labeling Step...")

    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="tactician labeling step execution",
    )
    async def execute(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute tactician model labeling."""
        try:
            self.logger.info("🔄 Executing Tactician Labeling...")

            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Use data sharing manager to get comprehensive data for tactician labeling
            self.logger.info(
                "🔄 Loading unified data for tactician labeling via data sharing manager...",
            )
            data_sharing_manager = get_data_sharing_manager(self.config)
            timeframe = training_input.get("timeframe", "1m")

            # Load unified data with optimizations for ML training
            # Use data sharing manager to avoid redundant loading
            from src.config.constants import (
                BLANK_TRAINING_LOOKBACK_DAYS,
            )

            # Use lookback_days from config (should be passed from enhanced training manager)
            config_lookback = self.config.get(
                "lookback_days", BLANK_TRAINING_LOOKBACK_DAYS,
            )
            data_1m = await data_sharing_manager.get_unified_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                lookback_days=config_lookback,
                force_reload=False,  # Use cache if available from previous steps,
            )

            if data_1m is None or data_1m.empty:
                self.logger.error(
                    f"🚨 No unified data found for {symbol} on {exchange}",
                )
                return {
                    "status": "FAILED",
                    "error": f"No unified data found for {symbol} on {exchange}",
                }

            # Log data information
            try:
                _loader = get_unified_data_loader(self.config)
                data_info = _loader.get_data_info(data_1m)
            except Exception as e:  # pragma: no cover - best effort logging
                self.logger.warning(f"⚠️ Could not get data info: {e}")
                data_info = {
                    "rows": len(data_1m) if hasattr(data_1m, "__len__") else None,
                    "columns": list(getattr(data_1m, "columns", [])) if hasattr(data_1m, "columns") else None,
                    "date_range": {"start": None, "end": None},
                    "has_aggtrades_data": False,
                    "has_futures_data": False,
                }
            self.logger.info(f"✅ Loaded unified data: {data_info['rows']} rows")
            with contextlib.suppress(Exception):
                self.logger.info(
                    f"   Date range: {data_info['date_range']['start']} to {data_info['date_range']['end']}",
                )
                self.logger.info(
                    f"   Has aggtrades data: {data_info['has_aggtrades_data']}",
                )
                self.logger.info(f"   Has futures data: {data_info['has_futures_data']}")

            # Ensure we have the required OHLCV columns
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in data_1m.columns
            ]
            if missing_columns:
                self.logger.error(f"🚨 Missing required columns: {missing_columns}")
                return {
                    "status": "FAILED",
                    "error": f"Missing required columns: {missing_columns}",
                }
            with contextlib.suppress(Exception):
                self.logger.info(
                    f"Loaded 1m data: shape={getattr(data_1m, 'shape', None)}, columns={list(getattr(data_1m, 'columns', [])[:10])}"
                )

            # Load analyst ensemble models
            analyst_ensembles = self._load_analyst_ensembles(data_dir)

            # Generate strategic "setup" signals using analyst models
            (
                data_with_features,
                strategic_signals,
            ) = await self._generate_strategic_signals(data_1m, analyst_ensembles)

            # Apply the specialized Tactician Triple Barrier
            labeler = TacticianTripleBarrierLabeler(self.config)
            labeled_data = labeler.apply_labels(data_with_features, strategic_signals)
            with contextlib.suppress(Exception):
                self.logger.info(
                    f"Strategic signals summary: total={len(strategic_signals)}, nonzero={(strategic_signals != 0).sum()}"
                )

            # Save results
            labeled_file, signals_file = self._save_results(
                labeled_data,
                strategic_signals,
                data_dir,
                exchange,
                symbol,
            )

            self.logger.info(
                f"✅ Tactician labeling completed. Labeled data saved to {labeled_file}",
            )

            pipeline_state["tactician_labeled_data"] = labeled_data
            return {
                "status": "SUCCESS",
                "labeled_file": labeled_file,
                "signals_file": signals_file,
            }
        except Exception as e:  # pragma: no cover - defensive
            self.logger.exception(f"❌ Error in Tactician Labeling: {e}")
            return {"status": "FAILED", "error": str(e)}

    def _load_analyst_ensembles(self, data_dir: str) -> dict[str, Any]:
        """Loads all trained analyst ensemble models."""
        analyst_ensembles_dir = f"{data_dir}/analyst_ensembles"
        analyst_ensembles: dict[str, Any] = {}
        if not Path(analyst_ensembles_dir).exists():
            msg = f"Analyst ensembles directory not found: {analyst_ensembles_dir}"
            raise FileNotFoundError(
                msg,
            )

        for ensemble_file in os.listdir(analyst_ensembles_dir):
            if ensemble_file.endswith("_ensemble.pkl"):
                regime_name = ensemble_file.replace("_ensemble.pkl", "")
                ensemble_path = Path(analyst_ensembles_dir) / ensemble_file
                with ensemble_path.open("rb") as f:
                    loaded = pickle.load(f)
                chosen_ensemble: Any = None
                if isinstance(loaded, dict):
                    # Prefer stacking_cv, then dynamic_weighting, then voting
                    for key in ENSEMBLE_PREFERENCE_ORDER:
                        if key in loaded and isinstance(loaded[key], dict):
                            obj = loaded[key].get("ensemble")
                            if obj is not None:
                                chosen_ensemble = obj
                                break
                    if chosen_ensemble is None:
                        # Fallback if saved dict is a single-ensemble payload
                        chosen_ensemble = (
                            loaded.get("ensemble") if "ensemble" in loaded else None
                        )
                # Record whatever we found (could be None; upstream handles None)
                analyst_ensembles[regime_name] = chosen_ensemble
        return analyst_ensembles

    async def _generate_strategic_signals(
        self, data: pd.DataFrame, analyst_ensembles: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate strategic signals using analyst ensemble models."""
        self.logger.info("Generating strategic 'setup' signals from Analyst models...")

        # Step 1: Calculate all features needed for any of the analyst models
        data_with_features = self._calculate_features(data)

        # Step 2: Determine the market regime for each data point
        # This is a placeholder for your regime detection logic (e.g., from step 4)
        # It is crucial that this logic is consistent with how regimes were defined during Analyst training.
        data_with_features["regime"] = self._get_market_regime(data_with_features)

        all_signals = pd.Series(0, index=data_with_features.index)

        # Step 3: Predict in a vectorized way for each regime
        for regime_name, ensemble in analyst_ensembles.items():
            if ensemble is None:
                continue

            regime_mask = data_with_features["regime"] == regime_name
            if not regime_mask.any():
                continue

            # Ensure the model's expected features are present
            if hasattr(ensemble, "feature_names_in_"):
                features_for_model = [
                    f
                    for f in getattr(ensemble, "feature_names_in_", [])
                    if f in data_with_features.columns
                ]
                x_regime = data_with_features.loc[regime_mask, features_for_model]
            else:
                # Fallback if feature names are not stored in the model
                x_regime = data_with_features.loc[regime_mask].select_dtypes(
                    include=np.number,
                )

            if not x_regime.empty:
                predictions = ensemble.predict(x_regime)
                all_signals[regime_mask] = predictions

        self.logger.info(
            f"Generated strategic signals. Signal distribution:\n{all_signals.value_counts()}",
        )
        return data_with_features, all_signals

    def _get_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """Placeholder for your market regime detection logic.
        This should be consistent with the logic from step4_regime_specific_training.
        """
        # Example: Simple regime based on volatility percentile
        # NOTE: Volatility is calculated here because the Analyst models need it for regime detection.
        # It is NOT used by the Tactician's labeler.
        vol_percentile = data["volatility"].rank(pct=True)
        bins = [0, 0.33, 0.66, 1.0]
        labels = ["SIDEWAYS", "BULL", "BEAR"]
        regimes = pd.cut(vol_percentile, bins=bins, labels=labels, right=False)
        return regimes.astype(str).fillna("SIDEWAYS")

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all necessary features for both Analyst and Tactician."""
        data = data.copy()
        data["returns"] = data["close"].pct_change()
        # Volatility is calculated here for the Analyst's regime detection, not for Tactician labeling.
        data["volatility"] = (
            data["returns"].rolling(window=60).std().bfill()
        )  # 1-hour volatility
        # ... Add all other features your Analyst models were trained on ...
        # e.g., RSI, MACD, Bollinger Bands, etc.
        return data.fillna(method="ffill").fillna(0)

    def _save_results(self, labeled_data: pd.DataFrame, signals: pd.Series, data_dir: str, exchange: str, symbol: str) -> Tuple[str, str]:
        """Saves the labeled data and signals to disk."""
        labeled_data_dir = f"{data_dir}/tactician_labeled_data"
        Path(labeled_data_dir).mkdir(parents=True, exist_ok=True)

        # Prefer Parquet for DataFrame/Series persistence
        labeled_file_parquet = (
            f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.parquet"
        )
        try:
            try:
                from src.training.enhanced_training_manager_optimized import (
                    ParquetDatasetManager,
                )

                ParquetDatasetManager(logger=self.logger).write_flat_parquet(
                    labeled_data,
                    labeled_file_parquet,
                    schema_name="split",
                    compression="snappy",
                    use_dictionary=True,
                    row_group_size=128_000,
                )
            except Exception:
                from src.utils.logger import log_dataframe_overview, log_io_operation

                with log_io_operation(
                    self.logger,
                    "to_parquet",
                    labeled_file_parquet,
                    compression="snappy",
                ):
                    labeled_data.to_parquet(
                        labeled_file_parquet, compression="snappy", index=False
                    )
                with contextlib.suppress(Exception):
                    log_dataframe_overview(
                        self.logger, labeled_data, name="labeled_data"
                    )
        except Exception:
            # Fallback to Pickle for compatibility
            labeled_file_pickle = (
                f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl"
            )
            labeled_data.to_pickle(labeled_file_pickle)
            labeled_file_parquet = labeled_file_pickle

        signals_file_parquet = (
            f"{data_dir}/{exchange}_{symbol}_strategic_signals.parquet"
        )
        try:
            # Save Series as Parquet by converting to DataFrame
            _signals_df = signals.to_frame(name="signal").reset_index()
            try:
                from src.training.enhanced_training_manager_optimized import (
                    ParquetDatasetManager,
                )

                ParquetDatasetManager(logger=self.logger).write_flat_parquet(
                    _signals_df,
                    signals_file_parquet,
                    schema_name="split",
                    compression="snappy",
                    use_dictionary=True,
                    row_group_size=128_000,
                )
            except Exception:
                from src.utils.logger import log_dataframe_overview, log_io_operation

                with log_io_operation(
                    self.logger,
                    "to_parquet",
                    signals_file_parquet,
                    compression="snappy",
                ):
                    _signals_df.to_parquet(
                        signals_file_parquet, compression="snappy", index=False
                    )
                with contextlib.suppress(Exception):
                    log_dataframe_overview(self.logger, _signals_df, name="signals_df")
        except Exception:
            signals_file_pickle = (
                f"{data_dir}/{exchange}_{symbol}_strategic_signals.pkl"
            )
            signals.to_pickle(signals_file_pickle)
            signals_file_parquet = signals_file_pickle

        return labeled_file_parquet, signals_file_parquet


# Import training pipeline decorators for comprehensive security and troubleshooting
from src.utils.centralized_decorators import (

from src.utils.enhanced_mlflow_integration import (
    with_enhanced_mlflow_logging,
    log_step_report,
    create_detailed_step_report,
    log_step_metrics,
    log_step_dataframe_with_standardized_name,
    log_step_artifact_with_standardized_name
)
    artifact_versioning,
    artifact_write_lock,
    circuit_breaker_protection,
    debug_training_step,
    deterministic_seed,
    idempotent_step,
    memory_efficient,
    nan_inf_and_constant_guard,
    prevent_data_leakage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    time_budget_watchdog,
    validate_step_output,
    validate_step_prerequisites,
)


# For backward compatibility with existing step structure
@deterministic_seed(42)
@idempotent_step(step_key="step8_tactician_labeling")
@artifact_write_lock()
@nan_inf_and_constant_guard()
@artifact_versioning("1.0")
@time_budget_watchdog(soft_timeout_seconds=2400.0)
@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=4.0,
    min_disk_gb=3.0,
    required_packages=["pandas", "numpy", "sklearn"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
    },
    context="Tactician Labeling",
)
@secure_data_processing(
    backup_before=True, integrity_checks=True, memory_cleanup=True, data_validation=True,
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=8.0,
    cpu_threshold_percent=80.0,
    disk_threshold_gb=5.0,
    monitor_interval=30.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=20000, streaming_processing=True, memory_pool=True, cleanup_frequency=40,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
    error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=120.0,
    expected_exception=Exception,
    monitor_interval=30.0,
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_tactician_labels.parquet"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["timestamp", "label", "signal"],
    },
    performance_thresholds={"labeling_time_minutes": 45.0},
    format_validation=True,
)
@quality_gate(
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"labeling_accuracy": 0.7},
)
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """Run the tactician labeling step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create step instance
        config: dict[str, Any] = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = TacticianLabelingStep(config)
        await step.initialize()

        # Execute step
        training_input: dict[str, Any] = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
            **kwargs,
        }

        pipeline_state: dict[str, Any] = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS"

    except Exception:  # pragma: no cover - defensive
        return False


if __name__ == "__main__":
    # Test the step
    async def test() -> None:
        await run_step("ETHUSDT", "BINANCE", "data/training")

    asyncio.run(test())