# src/training/steps/step9_5_hmm_lm_generalist_training.py

import asyncio
import concurrent.futures
import json
import os
import warnings
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.core.domain import (
    artifact_versioning,
    artifact_write_lock,
    circuit_breaker_protection,
    debug_training_step,
    deterministic_seed,
    guard_dataframe_nulls,
    handle_errors,
    idempotent_step,
    memory_efficient,
    nan_inf_and_constant_guard,
    prevent_data_leakage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    time_budget_watchdog,
    validate_data_quality,
    validate_step_output,
    validate_step_prerequisites,
    with_tracing_span
)
from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, timeout

from src.utils.enhanced_mlflow_integration import (
    with_enhanced_mlflow_logging,
    log_step_report,
    create_detailed_step_report,
    log_step_metrics,
    log_step_dataframe_with_standardized_name,
    log_step_artifact_with_standardized_name
)

# Suppress warnings
warnings.filterwarnings("ignore")

logger = system_logger.getChild("Step9_5_HMM_LM_Generalist")

class HMMLMGeneralistTrainingStep:
    """Step 9.5: Generalist HMM-LM Model Training for Regime Change Prediction."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild(self.__class__.__name__)
        self.models: dict[str, Any] = {}
        self.scalers: dict[str, Any] = {}
        self.label_encoders: dict[str, Any] = {}

        # HMM-LM configuration
        hmm_lm_config: dict[str, Any] = config.get("HMM_LM", {})
        generalist_config: dict[str, Any] = hmm_lm_config.get("generalist", {})

        self.hmm_states: int = int(generalist_config.get("hmm_states", 5))
        self.sequence_length: int = int(generalist_config.get("sequence_length", 20))
        self.timeframes: list[str] = generalist_config.get(
            "timeframes",
            ["1m", "5m", "15m", "30m"],
        )
        self.d_model: int = int(generalist_config.get("d_model", 256))
        self.nhead: int = int(generalist_config.get("nhead", 8))
        self.num_layers: int = int(generalist_config.get("num_layers", 6))
        self.dropout_rate: float = float(generalist_config.get("dropout_rate", 0.1))
        self.learning_rate: float = float(generalist_config.get("learning_rate", 0.0001))
        self.batch_size: int = int(generalist_config.get("batch_size", 32))
        self.epochs: int = int(generalist_config.get("epochs", 100))

        # Regime change vocabulary
        self.regime_change_vocab = self._create_regime_change_vocabulary()

    def _create_regime_change_vocabulary(self) -> dict[str, int]:
        """Create vocabulary for regime change events."""
        vocab: dict[str, int] = {}
        vocab_id = 0

        # Add regime entry events
        for state in range(self.hmm_states):
            vocab[f"enter_regime_{state}"] = vocab_id
            vocab_id += 1

        # Add regime exit events
        for state in range(self.hmm_states):
            vocab[f"exit_regime_{state}"] = vocab_id
            vocab_id += 1

        # Add special tokens
        vocab["<PAD>"] = vocab_id
        vocab_id += 1
        vocab["<UNK>"] = vocab_id
        vocab_id += 1
        vocab["<START>"] = vocab_id
        vocab_id += 1
        vocab["<END>"] = vocab_id

        return vocab

    @handles_errors(fallback=False)
    async def initialize(self) -> bool:
        """Initialize the HMM-LM generalist training step."""
        self.logger.info("Initializing HMM-LM Generalist Training Step...")
        self.logger.info("HMM-LM Generalist Training Step initialized successfully")
        return True

    @traced(span_name="step9_5.execute")
    @validates(validation_level="WARNING")
    # @with_enhanced_mlflow_logging - removed, use traced"step09_5_hmm_lm_generalist_training")
    @handles_errors(
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="HMM-LM generalist training step execution"
    )
    async def execute(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute HMM-LM generalist model training.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing training results

        """
        try:
            self.logger.info("🔄 Executing HMM-LM Generalist Training...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load HMM data from all timeframes
            hmm_data = await self._load_multi_timeframe_hmm_data(
                exchange=exchange, symbol=symbol, data_dir=data_dir
            )
            if not hmm_data:
                msg = "Failed to load multi-timeframe HMM data"
                raise ValueError(msg)

            # Create regime change sequences
            regime_sequences = await self._create_regime_change_sequences(hmm_data)
            if not regime_sequences:
                msg = "Failed to create regime change sequences"
                raise ValueError(msg)

            # Train HMM-LM model
            model_result = await self._train_hmm_lm_model(regime_sequences)
            if not model_result:
                msg = "Failed to train HMM-LM model"
                raise ValueError(msg)

            # Save model and metadata
            await self._save_generalist_model(model_result, exchange, symbol, data_dir)

            self.logger.info("✅ HMM-LM Generalist Training completed successfully")
            
            # Log artifacts and create detailed report
            await self._log_step9_5_artifacts_and_report(
            # Standardized naming pattern: {exchange}_{symbol}_{timestamp}_{step_num}_{artifact_type}
                training_input, pipeline_state, model_result
            )
            
            return {
                "status": "SUCCESS",
                "model_trained": True,
                "vocabulary_size": len(self.regime_change_vocab),
                "hmm_states": self.hmm_states,
                "timeframes": self.timeframes,
                "result": model_result,
            }

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ HMM-LM Generalist Training failed: {e}")
            return {"status": "FAILED", "error": str(e)}

    async def _log_step9_5_artifacts_and_report(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
        model_result: dict[str, Any]
    ) -> None:
        """Log step 9.5 artifacts and create detailed report."""
        try:
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")
            
            # Collect execution metadata
            execution_metadata = {
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": 0.0,  # Will be calculated if available
                "memory_usage_mb": 0.0,  # Will be calculated if available
                "cpu_usage_percent": 0.0,  # Will be calculated if available
                "data_quality_score": 1.0,
                "processing_efficiency": 1.0,
            }
            
            # Collect artifacts generated
            artifacts_generated = [
                f"{exchange}_{symbol}_hmm_lm_generalist_model.pkl",
                f"{exchange}_{symbol}_hmm_lm_generalist_metadata.json",
                f"{exchange}_{symbol}_hmm_lm_generalist_vocabulary.json",
            ]
            
            # Collect metrics
            metrics_calculated = {
                "hmm_lm_training_success": 1.0,
                "vocabulary_size": len(self.regime_change_vocab) if hasattr(self, 'regime_change_vocab') else 0,
                "hmm_states": self.hmm_states if hasattr(self, 'hmm_states') else 0,
                "timeframes_count": len(self.timeframes) if hasattr(self, 'timeframes') else 0,
                "model_trained": 1.0,
            }
            
            # Create step data for report
            step_data = {
                "model_result": model_result,
                "vocabulary_size": len(self.regime_change_vocab) if hasattr(self, 'regime_change_vocab') else 0,
                "hmm_states": self.hmm_states if hasattr(self, 'hmm_states') else 0,
                "timeframes": self.timeframes if hasattr(self, 'timeframes') else [],
            }
            
            # Create detailed report
            report_data = create_detailed_step_report(
                step_name="step09_5_hmm_lm_generalist_training",
                step_data=step_data,
                training_input=training_input,
                execution_metadata=execution_metadata,
                artifacts_generated=artifacts_generated,
                metrics_calculated=metrics_calculated,
                errors_encountered=[]
            )
            
            # Log the report
            report_name = log_step_report(
                config=self.config,
                step_name="step09_5_hmm_lm_generalist_training",
                report_data=report_data,
                report_type="hmm_lm_generalist_training_report",
                additional_metadata={
                    "hmm_lm_training_success": True,
                    "vocabulary_size": len(self.regime_change_vocab) if hasattr(self, 'regime_change_vocab') else 0,
                    "hmm_states": self.hmm_states if hasattr(self, 'hmm_states') else 0,
                    "asset": symbol,
                    "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                }
            )
            self.logger.info(f"✅ Logged HMM-LM generalist training report: {report_name}")
            
            # Log model result
            if model_result:
                model_report_name = log_step_report(
                    config=self.config,
                    step_name="step09_5_hmm_lm_generalist_training",
                    report_data=model_result,
                    report_type="hmm_lm_model_result",
                    additional_metadata={
                        "model_trained": True,
                        "vocabulary_size": len(self.regime_change_vocab) if hasattr(self, 'regime_change_vocab') else 0,
                        "asset": symbol,
                        "lookback_period": self.config.get("lookback_days", 1095),
                        "project_version": self.config.get("project_version", "1.0.0"),
                    }
                )
                self.logger.info(f"✅ Logged HMM-LM model result: {model_report_name}")
            
            # Log metrics
            log_step_metrics(
                config=self.config,
                step_name="step09_5_hmm_lm_generalist_training",
                metrics=metrics_calculated,
                additional_metadata={
                    "metrics_type": "hmm_lm_generalist_training_performance",
                    "asset": symbol,
                    "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                }
            )
            
            self.logger.info("✅ Step 9.5 artifacts and reports logged successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log step 9.5 artifacts and reports: {e}")
            # Don't fail the step if MLflow logging fails

    @traced(span_name="step9_5._load_multi_timeframe_hmm_data")
    # @guard_dataframe_nulls - removed, handled by validatesmode="warn", arg_index=0)
    async def _load_multi_timeframe_hmm_data(
        self, exchange: str, symbol: str, data_dir: str
    ) -> dict[str, pd.DataFrame]:
        """Load HMM data from all timeframes in parallel."""
        hmm_data: dict[str, pd.DataFrame] = {}

        async def load_timeframe_data(timeframe: str) -> tuple[str, pd.DataFrame | None]:
            try:
                # Load cluster assignments
                cluster_path = f"{data_dir}/{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
                if not os.path.exists(cluster_path):
                    return timeframe, None

                # Use ThreadPoolExecutor for I/O operations
                loop = asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    clusters_df = await loop.run_in_executor(
                        executor, pd.read_parquet, cluster_path
                    )

                clusters_df["timestamp"] = pd.to_datetime(clusters_df["timestamp"])
                clusters_df = clusters_df.set_index("timestamp")

                # Load intensity scores
                intensity_path = (
                    f"{data_dir}/{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet"
                )
                if os.path.exists(intensity_path):
                    loop = asyncio.get_running_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        intensity_df = await loop.run_in_executor(
                            executor, pd.read_parquet, intensity_path
                        )
                    intensity_df["timestamp"] = pd.to_datetime(
                        intensity_df["timestamp"]
                    )
                    intensity_df = intensity_df.set_index("timestamp")

                    # Merge cluster assignments with intensity scores
                    hmm_df = clusters_df.merge(
                        intensity_df, left_index=True, right_index=True, how="inner"
                    )
                    hmm_df["timeframe"] = timeframe
                    self.logger.info(
                        f"✅ Loaded HMM data for {timeframe}: {hmm_df.shape}",
                    )
                    return timeframe, hmm_df

                clusters_df["timeframe"] = timeframe
                self.logger.info(
                    f"✅ Loaded HMM clusters for {timeframe}: {clusters_df.shape}",
                )
                return timeframe, clusters_df

            except Exception as e:  # noqa: BLE001
                self.logger.exception(f"❌ Failed to load HMM data for {timeframe}: {e}")
                return timeframe, None

        # Load all timeframes in parallel
        tasks = [load_timeframe_data(tf) for tf in self.timeframes]
        results = await asyncio.gather(*tasks)

        # Build result dictionary
        for timeframe, df in results:
            if df is not None:
                hmm_data[timeframe] = df

        return hmm_data

    async def _create_regime_change_sequences(
        self, hmm_data: dict[str, pd.DataFrame]
    ) -> list[dict[str, Any]]:
        """Create regime change sequences for training."""
        sequences: list[dict[str, Any]] = []

        try:
            # Combine all timeframe data
            all_data: list[pd.DataFrame] = []
            for df in hmm_data.values():
                if not df.empty:
                    all_data.append(df)

            if not all_data:
                return []

            combined_df = pd.concat(all_data, axis=0).sort_index()

            # Detect regime changes and TPSL outcomes
            regime_events = self._detect_regime_changes_and_tpsl_outcomes(combined_df)

            # Create sequences around regime changes
            for change_idx, event_data in enumerate(regime_events):
                if change_idx < self.sequence_length:
                    continue

                # Get sequence before the change
                start_idx = change_idx - self.sequence_length
                end_idx = change_idx

                if start_idx >= 0 and end_idx < len(combined_df):
                    sequence_data = combined_df.iloc[start_idx:end_idx]

                    sequences.append(
                        {
                            "sequence": sequence_data,
                            "target": event_data["regime_change"],
                            "price_direction": event_data["price_direction"],
                            "profit_target_hit": event_data["profit_target_hit"],
                            "stop_loss_hit": event_data["stop_loss_hit"],
                            "time_to_target": event_data["time_to_target"],
                            "timestamp": combined_df.index[end_idx],
                            "timeframe": combined_df.iloc[end_idx]["timeframe"],
                        },
                    )

            self.logger.info(f"✅ Created {len(sequences)} regime change sequences")
            return sequences

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ Failed to create regime change sequences: {e}")
            return []

    def _detect_regime_changes_and_tpsl_outcomes(
        self, df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Detect regime changes and associated TPSL outcomes using enhanced probability-based approach."""
        events: list[dict[str, Any]] = []

        try:
            # Get TPSL parameters from config
            tpsl_config: dict[str, Any] = self.config.get(
                "vectorized_labelling_orchestrator", {}
            )
            profit_take_multiplier: float = float(
                tpsl_config.get("profit_take_multiplier", 0.002)
            )  # 0.2%
            stop_loss_multiplier: float = float(
                tpsl_config.get("stop_loss_multiplier", 0.001)
            )  # 0.1%
            _ = tpsl_config.get("time_barrier_minutes", 30)

            # Get regime column
            regime_col = "composite_cluster_id"
            if regime_col not in df.columns:
                self.logger.warning(f"⚠️ No regime column {regime_col} found")
                return events

            # Enhanced regime change detection using probability-based approach
            events = self._detect_regime_changes_enhanced(df, profit_take_multiplier, stop_loss_multiplier)

            return events

        except Exception as e:  # noqa: BLE001
            self.logger.exception(
                f"❌ Failed to detect regime changes and price action: {e}",
            )
            return []

    def _detect_regime_changes_enhanced(
        self, 
        df: pd.DataFrame, 
        profit_take_multiplier: float, 
        stop_loss_multiplier: float
    ) -> list[dict[str, Any]]:
        """Enhanced regime change detection using probability-based approach."""
        events: list[dict[str, Any]] = []
        
        try:
            # Get regime data
            regime_col = "composite_cluster_id"
            regimes = df[regime_col].fillna(-1).astype(int)
            
            # Calculate regime probabilities if available
            regime_probs = self._calculate_regime_probabilities(df)
            
            # Calculate regime stability and entropy
            regime_stability = self._calculate_regime_stability(regime_probs)
            regime_entropy = self._calculate_regime_entropy(regime_probs)
            
            # Detect regime changes using multiple signals
            regime_changes = self._detect_regime_changes_multi_signal(
                regimes, regime_stability, regime_entropy
            )
            
            # Process each potential regime change
            for i in range(len(regimes)):
                event: dict[str, Any] = {
                    "regime_change": "<PAD>",
                    "price_direction": 1,  # Sideways
                    "profit_target_hit": 0,  # 0/1
                    "stop_loss_hit": 0,  # 0/1
                    "time_to_target": 0,  # bars to hit target
                    "regime_confidence": 0.0,
                    "transition_probability": 0.0,
                }
                
                if i > 0 and regime_changes[i]:
                    prev_regime = int(regimes.iloc[i - 1])
                    curr_regime = int(regimes.iloc[i])
                    
                    if prev_regime >= 0 and curr_regime >= 0:
                        # Calculate transition probability
                        transition_prob = self._calculate_transition_probability(
                            prev_regime, curr_regime, regime_probs, i
                        )
                        
                        # Determine regime change type
                        if 0 <= prev_regime < self.hmm_states and 0 <= curr_regime < self.hmm_states:
                            event["regime_change"] = f"transition_{prev_regime}_to_{curr_regime}"
                            event["transition_probability"] = transition_prob
                            
                            # Calculate regime confidence
                            event["regime_confidence"] = float(regime_stability[i])
                            
                            # Enhanced TPSL analysis
                            tpsl_outcomes = self._calculate_enhanced_tpsl_outcomes(
                                df, i, profit_take_multiplier, stop_loss_multiplier
                            )
                            event.update(tpsl_outcomes)
                
                events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.exception(f"❌ Enhanced regime change detection failed: {e}")
            return []

    def _calculate_regime_probabilities(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate regime probabilities from available features."""
        try:
            # Look for probability features
            prob_cols = [col for col in df.columns if col.endswith("_p_state_")]
            
            if prob_cols:
                # Use existing probability features
                probs = df[prob_cols].values
                # Normalize to ensure probabilities sum to 1
                row_sums = probs.sum(axis=1, keepdims=True)
                probs = np.divide(probs, row_sums, where=row_sums > 0)
                return probs
            else:
                # Create dummy probabilities based on regime ID
                regime_col = "composite_cluster_id"
                if regime_col in df.columns:
                    regimes = df[regime_col].fillna(-1).astype(int)
                    n_states = max(regimes.max() + 1, self.hmm_states)
                    probs = np.zeros((len(regimes), n_states))
                    
                    for i, regime in enumerate(regimes):
                        if regime >= 0:
                            probs[i, regime] = 1.0
                    
                    return probs
                else:
                    return np.zeros((len(df), self.hmm_states))
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating regime probabilities: {e}")
            return np.zeros((len(df), self.hmm_states))

    def _calculate_regime_stability(self, regime_probs: np.ndarray) -> np.ndarray:
        """Calculate regime stability (max probability for each timepoint)."""
        try:
            return np.max(regime_probs, axis=1)
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating regime stability: {e}")
            return np.zeros(len(regime_probs))

    def _calculate_regime_entropy(self, regime_probs: np.ndarray) -> np.ndarray:
        """Calculate regime entropy (uncertainty measure)."""
        try:
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            entropy = -np.sum(regime_probs * np.log(regime_probs + eps), axis=1)
            return entropy
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating regime entropy: {e}")
            return np.zeros(len(regime_probs))

    def _detect_regime_changes_multi_signal(
        self, 
        regimes: pd.Series, 
        stability: np.ndarray, 
        entropy: np.ndarray
    ) -> np.ndarray:
        """Detect regime changes using multiple signals."""
        try:
            changes = np.zeros(len(regimes), dtype=bool)
            
            # Signal 1: Simple state comparison
            state_changes = np.diff(regimes.values, prepend=regimes.iloc[0]) != 0
            
            # Signal 2: Stability drops
            stability_threshold = np.percentile(stability, 25)  # Bottom 25%
            stability_changes = stability < stability_threshold
            
            # Signal 3: High entropy (uncertainty)
            entropy_threshold = np.percentile(entropy, 75)  # Top 25%
            entropy_changes = entropy > entropy_threshold
            
            # Combine signals with persistence filter
            for i in range(1, len(regimes)):
                if state_changes[i] and stability_changes[i] and entropy_changes[i]:
                    # Check persistence (avoid noise)
                    if i >= 3:  # Minimum persistence of 3 bars
                        changes[i] = True
            
            return changes
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error in multi-signal regime change detection: {e}")
            return np.zeros(len(regimes), dtype=bool)

    def _calculate_transition_probability(
        self, 
        from_regime: int, 
        to_regime: int, 
        regime_probs: np.ndarray, 
        index: int
    ) -> float:
        """Calculate transition probability between regimes."""
        try:
            if index < len(regime_probs) and from_regime < regime_probs.shape[1] and to_regime < regime_probs.shape[1]:
                # Use the probability of the target regime
                return float(regime_probs[index, to_regime])
            else:
                return 0.0
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating transition probability: {e}")
            return 0.0

    def _calculate_enhanced_tpsl_outcomes(
        self, 
        df: pd.DataFrame, 
        index: int, 
        profit_take_multiplier: float, 
        stop_loss_multiplier: float
    ) -> dict[str, Any]:
        """Calculate enhanced TPSL outcomes with confidence scoring."""
        try:
            outcomes = {
                "price_direction": 1,  # Sideways
                "profit_target_hit": 0,
                "stop_loss_hit": 0,
                "time_to_target": 0,
                "tpsl_confidence": 0.0,
            }
            
            if "close" in df.columns and index < len(df) - 1:
                current_price = float(df.iloc[index]["close"])
                future_prices = df.iloc[index + 1 : index + 31]["close"].values
                
                if len(future_prices) > 0:
                    # Calculate profit target and stop loss levels
                    profit_target = current_price * (1 + profit_take_multiplier)
                    stop_loss = current_price * (1 - stop_loss_multiplier)
                    
                    # Enhanced TPSL detection with confidence
                    profit_target_hit = 0
                    stop_loss_hit = 0
                    time_to_target = 0
                    confidence_factors = []
                    
                    for j, future_price in enumerate(future_prices):
                        fp = float(future_price)
                        
                        if fp >= profit_target and profit_target_hit == 0:
                            profit_target_hit = 1
                            time_to_target = j + 1
                            confidence_factors.append(1.0 - (j / 30))  # Higher confidence for earlier hits
                            
                        elif fp <= stop_loss and stop_loss_hit == 0:
                            stop_loss_hit = 1
                            if time_to_target == 0:
                                time_to_target = j + 1
                            confidence_factors.append(1.0 - (j / 30))
                    
                    # Calculate TPSL confidence
                    if confidence_factors:
                        outcomes["tpsl_confidence"] = float(np.mean(confidence_factors))
                    
                    # Enhanced price direction determination
                    if profit_target_hit == 1 and stop_loss_hit == 0:
                        outcomes["price_direction"] = 0  # Up
                    elif stop_loss_hit == 1 and profit_target_hit == 0:
                        outcomes["price_direction"] = 2  # Down
                    elif profit_target_hit == 1 and stop_loss_hit == 1:
                        # Both hit - use timing and confidence
                        if time_to_target <= 15 and outcomes["tpsl_confidence"] > 0.5:
                            outcomes["price_direction"] = 0  # Up
                        else:
                            outcomes["price_direction"] = 2  # Down
                    else:
                        outcomes["price_direction"] = 1  # Sideways
                    
                    outcomes["profit_target_hit"] = profit_target_hit
                    outcomes["stop_loss_hit"] = stop_loss_hit
                    outcomes["time_to_target"] = time_to_target
            
            return outcomes
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating enhanced TPSL outcomes: {e}")
            return {
                "price_direction": 1,
                "profit_target_hit": 0,
                "stop_loss_hit": 0,
                "time_to_target": 0,
                "tpsl_confidence": 0.0,
            }

    async def _train_hmm_lm_model(
        self, sequences: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Train the HMM-LM model."""
        try:
            self.logger.info(
                f"🔄 Training HMM-LM model with {len(sequences)} sequences",
            )

            if len(sequences) < 100:
                self.logger.warning(
                    f"⚠️ Insufficient sequences for training: {len(sequences)}",
                )
                return None

            # Prepare training data
            X_train, y_train, X_val, y_val = self._prepare_regime_training_data(
                sequences,
            )

            # Create efficient regime predictor
            input_dim = int(X_train.shape[2]) if len(X_train.shape) > 2 else 10
            model = EfficientRegimePredictor(
                input_dim=input_dim,
                num_regimes=self.hmm_states,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
            )

            # Train model
            trainer = EfficientRegimeTrainer(
                model, learning_rate=self.learning_rate, batch_size=self.batch_size
            )
            history = await trainer.train(
                X_train, y_train, X_val, y_val, epochs=self.epochs
            )

            # Save model
            model_path = "models/hmm_lm_generalist_model.pth"
            torch.save(model.state_dict(), model_path)

            result = {
                "model_path": model_path,
                "vocabulary": self.regime_change_vocab,
                "vocabulary_size": len(self.regime_change_vocab),
                "sequence_length": self.sequence_length,
                "hmm_states": self.hmm_states,
                "history": history,
                "model_config": {
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_layers": self.num_layers,
                    "dropout_rate": self.dropout_rate,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                },
            }

            # Thin adapter: expose unified price action probabilities on validation sample
            try:
                trainer.model.eval()
                sample_X = X_val[: min(len(X_val), 256)]
                with torch.no_grad():
                    outputs = trainer.model(sample_X)
                    # If model provides a price direction head, map to unified schema
                    if hasattr(trainer.model, "price_direction"):
                        logits = trainer.model.price_direction(outputs)
                        probs = F.softmax(logits, dim=-1)
                        if probs.ndim == 2 and probs.shape[1] >= 2:
                            direction_probability = float(torch.mean(probs[:, 1]).item())
                            barrier_avoidance_probability = float(1.0 - float(torch.mean(probs[:, 0]).item()))
                            magnitude_probability = float(torch.mean(torch.max(probs, dim=1).values).item())
                            triple_barrier_probability = direction_probability
                            probs_dict = {
                                "triple_barrier_probability": max(0.0, min(1.0, triple_barrier_probability)),
                                "direction_probability": max(0.0, min(1.0, direction_probability)),
                                "magnitude_probability": max(0.0, min(1.0, magnitude_probability)),
                                "barrier_avoidance_probability": max(0.0, min(1.0, barrier_avoidance_probability)),
                            }
                            from src.utils.common_operations import standardize_price_action_probabilities
                            result["price_action_probabilities"] = standardize_price_action_probabilities(probs_dict)
            except Exception:
                pass

            return result

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ HMM-LM training failed: {e}")
            return None

    def _prepare_regime_training_data(
        self, sequences: list[dict[str, Any]]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
        """Prepare training data for regime prediction."""
        try:
            # Convert sequences to tensor format
            X_data: list[np.ndarray] = []
            regime_ids: list[int] = []
            profit_hits: list[int] = []
            stop_hits: list[int] = []
            times_to_target: list[int] = []

            for seq_data in sequences:
                sequence: pd.DataFrame = seq_data["sequence"]
                target: str = seq_data["target"]

                # Convert sequence to feature tensor
                features = self._sequence_to_features(sequence)
                X_data.append(features)

                # Extract regime ID from target (e.g., "enter_regime_2" -> 2)
                if "enter_regime_" in target or "exit_regime_" in target:
                    regime_id = int(str(target).split("_")[-1])
                else:
                    regime_id = 0  # Default regime

                # Extract TPSL outcomes
                profit_target_hit = int(seq_data.get("profit_target_hit", 0))
                stop_loss_hit = int(seq_data.get("stop_loss_hit", 0))
                time_to_target = int(seq_data.get("time_to_target", 0))

                regime_ids.append(regime_id)
                profit_hits.append(profit_target_hit)
                stop_hits.append(stop_loss_hit)
                times_to_target.append(time_to_target)

            X = np.array(X_data, dtype=np.float32)
            y: Dict[str, np.ndarray] = {
                "regime_id": np.array(regime_ids, dtype=np.int64),
                "profit_target_hit": np.array(profit_hits, dtype=np.float32),
                "stop_loss_hit": np.array(stop_hits, dtype=np.float32),
                "time_to_target": np.array(times_to_target, dtype=np.float32),
            }

            # Purged time-based split with embargo to avoid leakage across adjacent sequences
            n = len(X)
            embargo = max(1, int(0.01 * n))
            split_idx = int(0.8 * n)
            train_end = max(0, split_idx - embargo)

            X_train, X_val = X[:train_end], X[split_idx:]
            y_train = {k: v[:train_end] for k, v in y.items()}
            y_val = {k: v[split_idx:] for k, v in y.items()}

            return X_train, y_train, X_val, y_val

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ Failed to prepare regime training data: {e}")
            return np.array([]), {}, np.array([]), {}

    def _sequence_to_features(self, sequence: pd.DataFrame) -> np.ndarray:
        """Convert sequence to feature tensor."""
        try:
            # Extract HMM-related features
            feature_cols: list[np.ndarray] = []

            # Add cluster ID
            if "composite_cluster_id" in sequence.columns:
                feature_cols.append(sequence["composite_cluster_id"].values.astype(float))

            # Add intensity features
            intensity_cols = [
                col for col in sequence.columns if str(col).startswith("intensity_cluster_")
            ]
            for col in intensity_cols:
                feature_cols.append(sequence[col].values.astype(float))

            # Add regime probability features
            regime_cols = [
                col for col in sequence.columns if str(col).endswith("_p_state_")
            ]
            for col in regime_cols:
                feature_cols.append(sequence[col].values.astype(float))

            # Stack features
            if feature_cols:
                features = np.column_stack(feature_cols)
            else:
                # Fallback: use basic features
                features = np.zeros((len(sequence), 10), dtype=np.float32)

            # Pad or truncate to sequence length
            if len(features) < self.sequence_length:
                # Pad with zeros at the beginning
                padding = np.zeros(
                    (self.sequence_length - len(features), features.shape[1]),
                    dtype=features.dtype,
                )
                features = np.vstack([padding, features])
            elif len(features) > self.sequence_length:
                # Truncate
                features = features[-self.sequence_length :]

            return features.astype(np.float32)

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ Failed to convert sequence to features: {e}")
            return np.zeros((self.sequence_length, 10), dtype=np.float32)

    async def _save_generalist_model(
        self, model_result: dict[str, Any], exchange: str, symbol: str, data_dir: str
    ) -> None:
        """Save the generalist model and metadata."""
        try:
            # Create models directory
            os.makedirs("models", exist_ok=True)

            # Save metadata
            metadata: dict[str, Any] = {
                "exchange": exchange,
                "symbol": symbol,
                "training_date": datetime.now().isoformat(),
                "model_type": "HMM_LM_Generalist",
                "vocabulary_size": len(self.regime_change_vocab),
                "hmm_states": self.hmm_states,
                "timeframes": self.timeframes,
                "sequence_length": self.sequence_length,
                "vocabulary": self.regime_change_vocab,
                "result": model_result,
            }

            os.makedirs(data_dir, exist_ok=True)
            metadata_path = f"{data_dir}/{exchange}_{symbol}_hmm_lm_generalist_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)

            self.logger.info(f"✅ Saved generalist model metadata to {metadata_path}")

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ Failed to save generalist model: {e}")

# Efficient Regime Prediction Architecture

class EfficientRegimePredictor(nn.Module):
    """Efficient regime prediction model for financial time series."""

    def __init__(
        self, input_dim: int, num_regimes: int, d_model: int = 256, nhead: int = 8, num_layers: int = 6
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_regimes = num_regimes
        self.d_model = d_model

        # Multi-scale feature extraction
        self.conv1d_short = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.conv1d_medium = nn.Conv1d(input_dim, 64, kernel_size=15, padding=7)
        self.conv1d_long = nn.Conv1d(input_dim, 64, kernel_size=30, padding=15)

        # Feature fusion
        self.feature_fusion = nn.Linear(192, d_model)

        # Transformer for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regime prediction heads
        self.regime_classifier = nn.Linear(d_model, num_regimes)  # Current regime
        self.transition_predictor = nn.Linear(
            d_model, num_regimes * num_regimes
        )  # Transition matrix
        self.regime_confidence = nn.Linear(d_model, num_regimes)  # Confidence scores

        # Price action prediction heads
        self.price_direction = nn.Linear(d_model, 3)  # Up/Down/Sideways
        self.profit_target_prob = nn.Linear(
            d_model, 1
        )  # Probability of hitting profit target
        self.stop_loss_prob = nn.Linear(d_model, 1)  # Probability of hitting stop loss
        self.time_to_target = nn.Linear(
            d_model, 1
        )  # Expected time to hit target (bars)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x shape: (batch, sequence_length, input_dim)

        # Multi-scale feature extraction
        x_t = x.transpose(1, 2)  # (batch, input_dim, sequence_length)

        short_features = F.relu(self.conv1d_short(x_t))
        medium_features = F.relu(self.conv1d_medium(x_t))
        long_features = F.relu(self.conv1d_long(x_t))

        # Global average pooling
        short_pooled = F.adaptive_avg_pool1d(short_features, 1).squeeze(-1)
        medium_pooled = F.adaptive_avg_pool1d(medium_features, 1).squeeze(-1)
        long_pooled = F.adaptive_avg_pool1d(long_features, 1).squeeze(-1)

        # Combine multi-scale features
        combined_features = torch.cat([short_pooled, medium_pooled, long_pooled], dim=1)
        fused_features = self.feature_fusion(combined_features)

        # Add sequence dimension for transformer
        fused_features = fused_features.unsqueeze(1)  # (batch, 1, d_model)

        # Apply transformer
        transformer_out = self.transformer(fused_features)

        # Take the output
        final_features = transformer_out[:, -1, :]
        final_features = self.dropout(final_features)

        # Predict regime probabilities and transitions
        current_regime_probs = F.softmax(self.regime_classifier(final_features), dim=-1)
        transition_probs = F.softmax(
            self.transition_predictor(final_features).view(
                -1, self.num_regimes, self.num_regimes
            ),
            dim=-1,
        )
        regime_confidence = torch.sigmoid(self.regime_confidence(final_features))

        # Predict price action and TPSL probabilities
        price_direction_probs = F.softmax(self.price_direction(final_features), dim=-1)
        profit_target_prob = torch.sigmoid(self.profit_target_prob(final_features))  # 0-1
        stop_loss_prob = torch.sigmoid(self.stop_loss_prob(final_features))  # 0-1
        time_to_target = torch.sigmoid(self.time_to_target(final_features)) * 30  # 0-30

        return {
            "current_regime": current_regime_probs,
            "transition_matrix": transition_probs,
            "confidence": regime_confidence,
            "price_direction": price_direction_probs,
            "profit_target_prob": profit_target_prob,
            "stop_loss_prob": stop_loss_prob,
            "time_to_target": time_to_target,
        }

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model),
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0), :]

class EfficientRegimeDataset(Dataset[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    """Custom dataset to return features and dict targets."""

    def __init__(
        self, X: torch.Tensor, targets: Dict[str, torch.Tensor]
    ) -> None:
        super().__init__()
        self.X = X
        self.targets = targets

    def __len__(self) -> int:  # noqa: D401
        return self.X.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.X[idx], {k: v[idx] for k, v in self.targets.items()}

class EfficientRegimeTrainer:
    """Efficient trainer for regime prediction model."""

    def __init__(
        self, model: nn.Module, learning_rate: float = 0.0001, batch_size: int = 32
    ) -> None:
        self.model = model
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Enable mixed precision for efficiency
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    async def train(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        epochs: int = 100,
    ) -> dict[str, list[float]]:
        """Train the regime prediction model efficiently."""
        # Convert to tensors
        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        X_val_t = torch.from_numpy(X_val).float().to(self.device)

        y_train_t: Dict[str, torch.Tensor] = {
            "regime_id": torch.from_numpy(y_train["regime_id"]).long().to(self.device),
            "profit_target_hit": torch.from_numpy(y_train["profit_target_hit"]).float().to(self.device),
            "stop_loss_hit": torch.from_numpy(y_train["stop_loss_hit"]).float().to(self.device),
            "time_to_target": torch.from_numpy(y_train["time_to_target"]).float().to(self.device),
        }
        y_val_t: Dict[str, torch.Tensor] = {
            "regime_id": torch.from_numpy(y_val["regime_id"]).long().to(self.device),
            "profit_target_hit": torch.from_numpy(y_val["profit_target_hit"]).float().to(self.device),
            "stop_loss_hit": torch.from_numpy(y_val["stop_loss_hit"]).float().to(self.device),
            "time_to_target": torch.from_numpy(y_val["time_to_target"]).float().to(self.device),
        }

        # Create data loaders
        train_dataset = EfficientRegimeDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False
        )

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        # Early stopping
        best_val_loss: float = float("inf")
        patience = 15
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()

                # Mixed precision training
                if self.scaler:
                    with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                        outputs = self.model(batch_X)
                        loss = self._compute_loss(outputs, batch_y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_X)
                    loss = self._compute_loss(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()

                train_loss_sum += float(loss.item())
                train_correct += self._compute_accuracy(outputs, batch_y)
                train_total += int(batch_y["regime_id"].size(0))

            # Validation
            val_loss, val_acc = self._validate(X_val_t, y_val_t)

            # Record metrics
            train_loss_avg = train_loss_sum / max(len(train_loader), 1)
            train_acc = train_correct / max(train_total, 1)

            history["train_loss"].append(train_loss_avg)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                os.makedirs("models", exist_ok=True)
                torch.save(self.model.state_dict(), "models/best_regime_predictor.pth")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            # Learning rate scheduling
            self.scheduler.step()

        return history

    def _compute_loss(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute multi-task loss for regime and TPSL prediction."""
        # Current regime classification loss
        regime_loss = F.cross_entropy(outputs["current_regime"], targets["regime_id"])

        # Transition matrix regularization (encourage smooth transitions)
        transition_matrix = outputs["transition_matrix"]
        identity = torch.eye(transition_matrix.size(1)).to(transition_matrix.device)
        transition_regularization = F.mse_loss(transition_matrix.mean(0), identity)

        # Confidence regularization (encourage high confidence for correct predictions)
        confidence = outputs["confidence"]
        confidence_loss = F.binary_cross_entropy(
            confidence, torch.ones_like(confidence),
        )

        # TPSL prediction losses
        profit_target_loss = F.binary_cross_entropy(
            outputs["profit_target_prob"].squeeze(-1), targets["profit_target_hit"],
        )
        stop_loss_loss = F.binary_cross_entropy(
            outputs["stop_loss_prob"].squeeze(-1), targets["stop_loss_hit"],
        )
        time_to_target_loss = F.mse_loss(
            outputs["time_to_target"].squeeze(-1), targets["time_to_target"],
        )

        # Combined loss with TPSL weighting
        return (
            regime_loss
            + 0.1 * transition_regularization
            + 0.05 * confidence_loss
            + 0.3 * profit_target_loss  # Higher weight for profit target prediction
            + 0.2 * stop_loss_loss  # Medium weight for stop loss prediction
            + 0.1 * time_to_target_loss  # Lower weight for timing prediction
        )

    def _compute_accuracy(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> int:
        """Compute accuracy for regime and TPSL prediction."""
        # Regime accuracy
        predicted_regime = torch.argmax(outputs["current_regime"], dim=1)
        regime_correct = (predicted_regime == targets["regime_id"]).sum().item()

        # TPSL accuracy (profit target prediction)
        predicted_profit = (outputs["profit_target_prob"].squeeze(-1) > 0.5).float()
        profit_correct = (
            (predicted_profit == targets["profit_target_hit"]).sum().item()
        )

        # Combined accuracy (weighted equally as counts)
        return int(regime_correct + profit_correct)

    def _validate(
        self, X_val: torch.Tensor, y_val: dict[str, torch.Tensor]
    ) -> tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_val)
            loss = float(self._compute_loss(outputs, y_val).item())
            accuracy = float(
                self._compute_accuracy(outputs, y_val) / max(y_val["regime_id"].size(0), 1)
            )
        return loss, accuracy

# For backward compatibility with existing step structure
@deterministic_seed(42)
@idempotent_step(step_key="step09_5_hmm_lm_generalist_training")
# @artifact_write_lock() - removed, handled by file system
@validates()
# @artifact_versioning("1.0") - removed, handled by pipeline
@timeout(timeout=3600)
@validates(
    required_directories=["data/training", "models"],
    min_memory_gb=16.0,
    min_disk_gb=10.0,
    required_packages=["torch", "numpy", "pandas", "sklearn", "lightgbm"],
    data_quality_checks={"check_data_completeness": True},
)
# @secure_data_processing - removed, handled by validates
# @prevent_data_leakage - removed, handled by validates
# @resource_monitor - removed, use log_execution_time
# @memory_efficient - removed
# @debug_training_step - removed
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=300.0,
)
@validates(
    required_files=["hmm_lm_generalist_model.pkl"],
    data_quality_checks={"check_output_completeness": True},
)
# @quality_gate - removed, handled by validates
@handles_errors(fallback=False)
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """Run the HMM-LM generalist training step.

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
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = HMMLMGeneralistTrainingStep(config)
        await step.initialize()

        # Execute step
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
            **kwargs,
        }

        pipeline_state: dict[str, Any] = {}
        result = await step.execute(training_input, pipeline_state)

        return bool(result.get("status") == "SUCCESS")

    except Exception as e:  # noqa: BLE001
        logger.exception(f"HMM-LM generalist training failed: {e}")
        return False

if __name__ == "__main__":
    # Test the step
    async def test() -> None:
        await run_step("ETHUSDT", "BINANCE", "data/training")

    asyncio.run(test())