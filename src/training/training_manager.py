# src/training/training_manager.py

import asyncio
import os
import json
import pickle
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import warnings
import time
import traceback

warnings.filterwarnings("ignore")
import mlflow

from src.utils.model_manager import ModelManager
from src.utils.logger import system_logger
from src.config import CONFIG
from src.database.sqlite_manager import SQLiteManager
from exchange.binance import BinanceExchange
from src.utils.error_handler import handle_errors
from src.analyst.analyst import Analyst
from src.supervisor.main import Supervisor
from src.utils.state_manager import StateManager

# Import the new RegularizationManager
from src.training.regularization import RegularizationManager


class TrainingManager:
    """
    Comprehensive training manager for the Ares Trading Bot.
    Orchestrates the entire training pipeline by leveraging existing components:
    - Analyst for feature engineering and model training
    - Supervisor for optimization and validation
    - Backtesting modules for walk-forward and Monte Carlo validation

    This manager provides a unified interface for:
    1. Full training pipeline for specific tokens
    2. Model retraining with latest data
    3. Model import/export functionality
    4. Training status and history tracking
    5. L1-L2 regularization configuration and enforcement
    """

    def __init__(self, db_manager: SQLiteManager):
        self.db_manager = db_manager
        self.logger = system_logger.getChild("TrainingManager")
        self.models_dir = "models"
        self.data_dir = "data/training"
        self.reports_dir = "reports"
        self.steps_dir = "src/training/steps"  # New directory for modular steps

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.steps_dir, exist_ok=True)  # Ensure steps directory exists

        self.state_manager = StateManager()
        self.model_manager = ModelManager()
        self.analyst = None
        self.supervisor = None

        self.current_training_session = None
        self.training_history = {}

        # Initialize RegularizationManager
        self.regularization_manager = RegularizationManager()

        # Set up MLflow
        self.mlflow_enabled = False
        if CONFIG.get("MLFLOW_TRACKING_URI"):
            try:
                mlflow.set_tracking_uri(CONFIG.get("MLFLOW_TRACKING_URI"))
                mlflow.set_experiment(CONFIG.get("MLFLOW_EXPERIMENT_NAME", "Ares_Training"))
                self.logger.info(
                    f"MLflow tracking URI set to: {CONFIG.get('MLFLOW_TRACKING_URI')}"
                )
                self.mlflow_enabled = True
            except Exception as e:
                self.logger.warning(f"Failed to set MLflow experiment: {e}")
                self.mlflow_enabled = False
        else:
            self.logger.warning(
                "MLFLOW_TRACKING_URI not set in config. MLflow tracking is disabled."
            )

    # Removed _get_regularization_config as it's now in RegularizationManager

    def apply_regularization_to_components(self):
        """Apply L1-L2 regularization configuration to all training components."""
        try:
            if self.analyst and hasattr(self.analyst, "predictive_ensembles"):
                # Delegate to RegularizationManager
                self.regularization_manager.apply_regularization_to_ensembles(
                    self.analyst.predictive_ensembles
                )
            else:
                self.logger.warning(
                    "Analyst or its predictive_ensembles not initialized. Cannot apply regularization."
                )

            self.logger.info(
                "Successfully applied regularization configuration to all components"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to apply regularization configuration: {e}", exc_info=True
            )

    # Removed _apply_regularization_to_ensemble as it's now a private helper in RegularizationManager

    def validate_and_report_regularization(self) -> bool:
        """
        Validate regularization configuration and report on the setup.
        Delegates to RegularizationManager.

        Returns:
            bool: True if regularization is properly configured, False otherwise
        """
        return self.regularization_manager.validate_and_report_regularization()

    async def initialize_components(self, symbol: str):
        """Initialize components for training. Simplified for blank training runs."""
        try:
            # For blank training runs, we don't need to initialize all components
            # Just ensure basic setup is complete
            self.logger.info(f"Basic component setup completed for {symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}", exc_info=True)
            return False

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="full_training_pipeline"
    )
    async def run_full_training(
        self, symbol: str, exchange_name: str = "BINANCE", timeframe: str = "1h", lookback_days_override: Optional[int] = None
    ) -> Optional[str]:
        """
        Orchestrates the complete training pipeline by calling modular scripts.
        """
        start_time = time.time()
        self.logger.info("=" * 80)
        self.logger.info("🚀 FULL TRAINING PIPELINE START")
        self.logger.info("=" * 80)
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Symbol: {symbol}")
        self.logger.info(f"Exchange: {exchange_name}")
        self.logger.info(f"Timeframe: {timeframe}")
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"Working directory: {os.getcwd()}")
        
        self.logger.info(
            f"Starting full training pipeline for {symbol} on {exchange_name}"
        )
        session_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=f"training_{session_id}") as run:
            run_id = run.info.run_id
            self.logger.info(f"MLflow run started. Run ID: {run_id}")

            self.current_training_session = {
                "session_id": session_id,
                "symbol": symbol,
                "exchange": exchange_name,
                "start_time": datetime.now().isoformat(),
                "status": "running",
                "mlflow_run_id": run_id,
            }

            try:
                # --- Initial Setup ---
                self.logger.info("🔧 Step 0: Initial setup and MLflow configuration...")
                setup_start_time = time.time()
                
                if self.mlflow_enabled:
                    mlflow.log_params(CONFIG.get("MODEL_TRAINING", {}))
                    mlflow.log_param("symbol", symbol)
                    mlflow.log_param("exchange", exchange_name)
                    mlflow.log_param("timeframe", timeframe)
                
                setup_duration = time.time() - setup_start_time
                self.logger.info(f"✅ Initial setup completed in {setup_duration:.2f} seconds")

                self.logger.info("🔧 Step 0.1: Initializing components...")
                components_start_time = time.time()
                
                if not await self.initialize_components(symbol):
                    self.logger.error("💥 Component initialization failed")
                    return None
                    
                components_duration = time.time() - components_start_time
                self.logger.info(f"✅ Component initialization completed in {components_duration:.2f} seconds")
                
                self.logger.info("🔧 Step 0.2: Applying regularization...")
                regularization_start_time = time.time()
                
                self.apply_regularization_to_components()  # This now calls RegularizationManager
                
                regularization_duration = time.time() - regularization_start_time
                self.logger.info(f"✅ Regularization applied in {regularization_duration:.2f} seconds")
                
                self.logger.info("🔧 Step 0.3: Validating regularization...")
                validation_start_time = time.time()
                
                if (
                    not self.validate_and_report_regularization()
                ):  # This now calls RegularizationManager
                    self.logger.warning(
                        "⚠️ Regularization validation failed, continuing training..."
                    )
                else:
                    self.logger.info("✅ Regularization validation passed")
                    
                validation_duration = time.time() - validation_start_time
                self.logger.info(f"✅ Regularization validation completed in {validation_duration:.2f} seconds")

                # --- Pipeline Step 1: Data Collection ---
                self.logger.info("🔧 Step 1: Collecting and preparing data...")
                step1_start_time = time.time()
                
                # Use the override for blank runs, otherwise use the config default for full runs.
                lookback_to_use = lookback_days_override if lookback_days_override is not None else CONFIG["MODEL_TRAINING"]["data_retention_days"]
                
                success = await self._run_step_script(
                    "step1_data_collection.py",
                    [
                        symbol,
                        exchange_name,
                        str(lookback_to_use),
                        str(CONFIG["MODEL_TRAINING"]["min_data_points"]),
                        self.data_dir,
                    ],
                )
                
                step1_duration = time.time() - step1_start_time
                self.logger.info(f"⏱️  Step 1 completed in {step1_duration:.2f} seconds")
                
                if not success:
                    self.logger.error("💥 Step 1 (Data Collection) failed")
                    return None
                else:
                    self.logger.info("✅ Step 1 (Data Collection) completed successfully")

                # Load data collected by step 1 (needed for subsequent steps)
                data_file = os.path.join(self.data_dir, f"{symbol}_historical_data.pkl")
                self.logger.info(f"📁 Data file path: {data_file}")
                
                # Check if data file exists
                if not os.path.exists(data_file):
                    self.logger.error(f"💥 Data file not found: {data_file}")
                    return None
                else:
                    self.logger.info(f"✅ Data file exists: {data_file}")
                
                # with open(data_file, "rb") as f:
                #     collected_data = pickle.load(f)
                # klines_df = collected_data["klines"]
                # agg_trades_df = collected_data["agg_trades"]
                # futures_df = collected_data["futures"]

                # --- Pipeline Step 2: Preliminary Optimization (Stage 1) ---
                self.logger.info("🔧 Step 2: Running Preliminary Target Parameter Optimization...")
                step2_start_time = time.time()
                
                success = await self._run_step_script(
                    "step2_preliminary_optimization.py",
                    [
                        symbol,
                        timeframe,
                        self.data_dir,
                        data_file,
                    ],  # Pass data_file path
                )
                
                step2_duration = time.time() - step2_start_time
                self.logger.info(f"⏱️  Step 2 completed in {step2_duration:.2f} seconds")
                
                if not success:
                    self.logger.error("💥 Step 2 (Preliminary Optimization) failed")
                    return None
                else:
                    self.logger.info("✅ Step 2 (Preliminary Optimization) completed successfully")
                    
                optimal_target_params = self._load_intermediate_result(
                    f"{symbol}_optimal_target_params.json"
                )
                if not optimal_target_params:
                    self.logger.error("💥 Failed to load optimal target parameters")
                    return None
                else:
                    self.logger.info(f"✅ Loaded optimal target parameters: {optimal_target_params}")
                    
                if self.mlflow_enabled:
                    mlflow.log_params(
                        {
                            "optimal_tp_threshold": optimal_target_params["tp_threshold"],
                            "optimal_sl_threshold": optimal_target_params["sl_threshold"],
                            "optimal_holding_period": optimal_target_params[
                                "holding_period"
                            ],
                        }
                    )

                # --- Pipeline Step 3: Coarse Optimization & Pruning (Stage 2) ---
                self.logger.info("🔧 Step 3: Running Coarse Optimization and Feature Pruning...")
                step3_start_time = time.time()
                
                success = await self._run_step_script(
                    "step3_coarse_optimization.py",
                    [
                        symbol,
                        timeframe,
                        self.data_dir,
                        data_file,
                        json.dumps(optimal_target_params),
                    ],
                )
                
                step3_duration = time.time() - step3_start_time
                self.logger.info(f"⏱️  Step 3 completed in {step3_duration:.2f} seconds")
                
                if not success:
                    self.logger.error("💥 Step 3 (Coarse Optimization) failed")
                    return None
                else:
                    self.logger.info("✅ Step 3 (Coarse Optimization) completed successfully")
                    
                pruned_features = self._load_intermediate_result(
                    f"{symbol}_pruned_features.json"
                )
                hpo_ranges = self._load_intermediate_result(f"{symbol}_hpo_ranges.json")
                if not pruned_features or not hpo_ranges:
                    self.logger.error("💥 Failed to load pruned features or HPO ranges")
                    return None
                else:
                    self.logger.info(f"✅ Loaded pruned features: {len(pruned_features)}")
                    self.logger.info(f"✅ Loaded HPO ranges: {len(hpo_ranges)}")
                    
                if self.mlflow_enabled:
                    mlflow.log_param("pruned_features_count", len(pruned_features))
                    mlflow.log_dict(hpo_ranges, "coarse_hpo_ranges")

                # --- Pipeline Step 4: Main Model Training (Stage 3a) ---
                self.logger.info("🔧 Step 4: Training main models with pruned features...")
                step4_start_time = time.time()
                
                success = await self._run_step_script(
                    "step4_main_model_training.py",
                    [
                        symbol,
                        timeframe,
                        self.data_dir,
                        data_file,
                        json.dumps(optimal_target_params),
                        json.dumps(pruned_features),
                    ],
                )
                
                step4_duration = time.time() - step4_start_time
                self.logger.info(f"⏱️  Step 4 completed in {step4_duration:.2f} seconds")
                
                if not success:
                    self.logger.error("💥 Step 4 (Main Model Training) failed")
                    return None
                else:
                    self.logger.info("✅ Step 4 (Main Model Training) completed successfully")

                # --- Pipeline Step 5: 4-Stage Hyperparameter Optimization ---
                self.logger.info(
                    "Step 5: Running 4-stage hyperparameter optimization (5, 20, 30, 50 trials)..."
                )
                success = await self._run_step_script(
                    "step5_multi_stage_hpo.py",
                    [symbol, self.data_dir, data_file],
                )
                if not success:
                    return None
                # The best_params will be updated directly in CONFIG by the optimizer, no need to load here.

                # --- Final Validation Steps ---
                self.logger.info("Step 6: Performing walk-forward validation...")
                success = await self._run_step_script(
                    "step6_walk_forward_validation.py",
                    [symbol, self.data_dir, data_file],
                )
                if not success:
                    return None
                wfa_metrics = self._load_intermediate_result(
                    f"{symbol}_wfa_metrics.json"
                )
                if wfa_metrics:
                    mlflow.log_metrics({"wfa_" + k: v for k, v in wfa_metrics.items()})
                else:
                    return None

                self.logger.info("Step 7: Performing Monte Carlo validation...")
                success = await self._run_step_script(
                    "step7_monte_carlo_validation.py",
                    [symbol, self.data_dir, data_file],
                )
                if not success:
                    return None
                mc_metrics = self._load_intermediate_result(f"{symbol}_mc_metrics.json")
                if mc_metrics:
                    mlflow.log_metrics({"mc_" + k: v for k, v in mc_metrics.items()})
                else:
                    return None

                self.logger.info("Step 8: Setting up A/B testing...")
                success = await self._run_step_script(
                    "step8_ab_testing_setup.py", [symbol]
                )
                if not success:
                    return None

                self.logger.info("Step 9: Saving training results and artifacts...")
                success = await self._run_step_script(
                    "step9_save_results.py",
                    [
                        symbol,
                        session_id,
                        run_id,
                        self.data_dir,
                        self.reports_dir,
                        self.models_dir,
                    ],
                )
                if not success:
                    return None

                mlflow.log_artifacts(self.reports_dir, artifact_path="reports")
                mlflow.log_artifacts(self.models_dir, artifact_path="models")

                self.current_training_session["status"] = "completed"
                self.current_training_session["end_time"] = datetime.now().isoformat()

                self.logger.info(
                    f"Full training pipeline completed successfully for {symbol}"
                )
                return run_id

            except Exception as e:
                self.logger.error(f"Full training pipeline failed: {e}", exc_info=True)
                self.current_training_session["status"] = "failed"
                self.current_training_session["error"] = str(e)
                mlflow.set_tag("status", "FAILED")
                return None

    async def _run_step_script(self, script_name: str, args: List[str]) -> bool:
        """Runs a training step script as a subprocess."""
        import os
        import asyncio

        script_path = os.path.join(self.steps_dir, script_name)
        command = [sys.executable, script_path] + args

        self.logger.info(f"🔧 Starting subprocess execution:")
        self.logger.info(f"   Script: {script_name}")
        self.logger.info(f"   Full path: {script_path}")
        self.logger.info(f"   Command: {' '.join(command)}")
        self.logger.info(f"   Arguments: {args}")
        self.logger.info(f"   Python executable: {sys.executable}")
        self.logger.info(f"   Steps directory: {self.steps_dir}")

        # Check if script exists
        if not os.path.exists(script_path):
            self.logger.error(f"💥 Script not found: {script_path}")
            self.logger.error(f"   Available files in {self.steps_dir}:")
            try:
                for file in os.listdir(self.steps_dir):
                    self.logger.error(f"     - {file}")
            except Exception as e:
                self.logger.error(f"     Error listing directory: {e}")
            return False

        # Set up environment
        env = os.environ.copy()
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(self.steps_dir)))
        env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"
        
        # Pass blank training mode flag to subprocess
        from src.config import CONFIG
        if CONFIG.get("BLANK_TRAINING_MODE", False):
            env['BLANK_TRAINING_MODE'] = '1'
            self.logger.info(f"🔧 Passing BLANK_TRAINING_MODE=1 to subprocess")
        else:
            env['BLANK_TRAINING_MODE'] = '0'
        
        self.logger.info(f"🔧 Environment setup:")
        self.logger.info(f"   Project root: {project_root}")
        self.logger.info(f"   PYTHONPATH: {env['PYTHONPATH']}")
        self.logger.info(f"   BLANK_TRAINING_MODE: {env.get('BLANK_TRAINING_MODE', '0')}")
        self.logger.info(f"   Working directory: {os.getcwd()}")

        self.logger.info(f"Running step: {script_name} with args: {args}")
        
        subprocess_start_time = time.time()
        try:
            self.logger.info(f"🔧 Creating subprocess...")
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=env
            )
            self.logger.info(f"✅ Subprocess created successfully. PID: {process.pid}")
            
            self.logger.info(f"🔧 Starting real-time subprocess monitoring...")
            self.logger.info(f"   Process PID: {process.pid}")
            communication_start_time = time.time()
            
            # Collect real-time output
            stdout_lines = []
            stderr_lines = []
            
            async def read_stream(stream, is_stderr=False):
                """Read from stream and log in real-time."""
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    line_text = line.decode().strip()
                    if line_text:
                        if is_stderr:
                            stderr_lines.append(line_text)
                            self.logger.warning(f"[SUBPROCESS-{process.pid}] {line_text}")
                        else:
                            stdout_lines.append(line_text)
                            self.logger.info(f"[SUBPROCESS-{process.pid}] {line_text}")
            
            try:
                self.logger.info(f"   Starting real-time output monitoring...")
                # Start reading from both streams concurrently
                await asyncio.gather(
                    read_stream(process.stdout, is_stderr=False),
                    read_stream(process.stderr, is_stderr=True)
                )
                
                # Wait for process to complete
                return_code = await process.wait()
                self.logger.info(f"   Subprocess completed with return code: {return_code}")
                
                communication_duration = time.time() - communication_start_time
                subprocess_duration = time.time() - subprocess_start_time
                
                self.logger.info(f"⏱️  Subprocess execution completed:")
                self.logger.info(f"   Communication duration: {communication_duration:.2f} seconds")
                self.logger.info(f"   Total subprocess duration: {subprocess_duration:.2f} seconds")
                self.logger.info(f"   Return code: {return_code}")
                self.logger.info(f"   STDOUT lines captured: {len(stdout_lines)}")
                self.logger.info(f"   STDERR lines captured: {len(stderr_lines)}")
                
            except asyncio.TimeoutError:
                communication_duration = time.time() - communication_start_time
                subprocess_duration = time.time() - subprocess_start_time
                self.logger.error(f"💥 Subprocess communication timed out after {communication_duration:.2f} seconds")
                self.logger.error(f"   Total subprocess duration: {subprocess_duration:.2f} seconds")
                self.logger.error(f"   Script: {script_name}")
                self.logger.error(f"   PID: {process.pid}")
                
                # Try to terminate the process
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=10)
                    self.logger.info("✅ Process terminated successfully")
                except asyncio.TimeoutError:
                    self.logger.error("💥 Process termination timed out, killing forcefully")
                    process.kill()
                    await process.wait()
                
                return False
                
            # Log summary of captured output
            self.logger.info(f"📊 Subprocess output summary:")
            self.logger.info(f"   STDOUT lines: {len(stdout_lines)}")
            self.logger.info(f"   STDERR lines: {len(stderr_lines)}")
            
            if stdout_lines:
                self.logger.info(f"   STDOUT content (first 10 lines):")
                for i, line in enumerate(stdout_lines[:10]):
                    self.logger.info(f"     [{i+1}] {line}")
                if len(stdout_lines) > 10:
                    self.logger.info(f"     ... and {len(stdout_lines) - 10} more lines")
                    
            if stderr_lines:
                self.logger.warning(f"   STDERR content (first 10 lines):")
                for i, line in enumerate(stderr_lines[:10]):
                    self.logger.warning(f"     [{i+1}] {line}")
                if len(stderr_lines) > 10:
                    self.logger.warning(f"     ... and {len(stderr_lines) - 10} more lines")

            if return_code != 0:
                self.logger.error(f"💥 Step '{script_name}' failed with exit code {return_code}.")
                self.logger.error(f"   STDOUT lines: {len(stdout_lines)}")
                self.logger.error(f"   STDERR lines: {len(stderr_lines)}")
                if stdout_lines:
                    self.logger.error(f"   STDOUT (last 5 lines):")
                    for line in stdout_lines[-5:]:
                        self.logger.error(f"     {line}")
                if stderr_lines:
                    self.logger.error(f"   STDERR (last 5 lines):")
                    for line in stderr_lines[-5:]:
                        self.logger.error(f"     {line}")
                return False
            else:
                self.logger.info(f"✅ Step '{script_name}' completed successfully.")
                if stdout_lines:
                    self.logger.debug(f"   STDOUT lines: {len(stdout_lines)}")
                return True
                
        except FileNotFoundError as e:
            self.logger.error(f"💥 FileNotFoundError running step '{script_name}': {e}")
            self.logger.error(f"   Script path: {script_path}")
            self.logger.error(f"   Command: {' '.join(command)}")
            self.logger.error(f"   Python executable: {sys.executable}")
            return False
            
        except asyncio.CancelledError as e:
            self.logger.error(f"💥 Subprocess cancelled: {e}")
            self.logger.error(f"   This usually indicates the process was interrupted (Ctrl+C)")
            self.logger.error(f"   Script: {script_name}")
            self.logger.error(f"   Duration before cancellation: {time.time() - subprocess_start_time:.2f} seconds")
            return False
            
        except Exception as e:
            self.logger.error(f"💥 Unexpected error running step '{script_name}': {e}")
            self.logger.error(f"   Error type: {type(e).__name__}")
            self.logger.error(f"   Script path: {script_path}")
            self.logger.error(f"   Command: {' '.join(command)}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            return False

    def _load_intermediate_result(self, filename: str) -> Any:
        """Helper to load intermediate results (JSON or pickle)."""
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            self.logger.error(f"Intermediate result file not found: {file_path}")
            return None
        try:
            if filename.endswith(".json"):
                with open(file_path, "r") as f:
                    return json.load(f)
            elif filename.endswith(".pkl"):
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            else:
                self.logger.error(f"Unsupported intermediate file type: {filename}")
                return None
        except Exception as e:
            self.logger.error(
                f"Error loading intermediate result from {file_path}: {e}",
                exc_info=True,
            )
            return None

    def _parse_report_for_metrics(self, report_content: str) -> Dict[str, float]:
        """Parses a text report to extract key-value metrics."""
        import re  # Import re here

        metrics = {}
        patterns = {
            "sharpe_ratio": r"Sharpe Ratio:\s*(-?\d+\.\d+)",
            "max_drawdown": r"Max Drawdown \(%\):\s*(-?\d+\.\d+)",  # Corrected pattern
            "win_rate": r"Win Rate \(%\):\s*(\d+\.\d+)",  # Corrected pattern
            "profit_factor": r"Profit Factor:\s*(\d+\.\d+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, report_content)
            if match:
                try:
                    # Remove commas from numbers if present (e.g., $10,000.00)
                    value = match.group(1).replace(",", "")
                    metrics[
                        key.replace(" ", "_").replace("%", "Pct").replace(".", "")
                    ] = float(value)
                except (ValueError, IndexError):
                    continue
        return metrics

    @handle_errors(
        exceptions=(Exception,), default_return=False, context="model_retraining"
    )
    @handle_errors(
        exceptions=(Exception,), default_return=False, context="model_retraining"
    )
    async def retrain_models(self, symbol: str, exchange_name: str = "BINANCE") -> bool:
        """Retrain models with latest data."""
        self.logger.info(f"Starting model retraining for {symbol}")
        return await self.run_full_training(symbol, exchange_name, timeframe="1m")

    @handle_errors(
        exceptions=(Exception,), default_return=False, context="model_import"
    )
    async def import_model(self, model_path: str, symbol: str) -> bool:
        """Import a trained model from file."""
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False

            # Load the model
            # with open(model_path, "rb") as f:
            #     # model_data = pickle.load(f)
            #     pass

            # Use Analyst's prediction functionality
            if self.analyst:
                prediction = await self.analyst.run_analysis_pipeline()
                return prediction

            return None

        except Exception as e:
            self.logger.error(f"Model import failed: {e}", exc_info=True)
            return False

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="model_prediction"
    )
    async def predict(
        self, features: pd.DataFrame, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Make predictions using the trained models for a symbol."""
        try:
            # Load the appropriate model for the symbol
            model_path = os.path.join(self.models_dir, f"{symbol}_best_model.pkl")
            if not os.path.exists(model_path):
                self.logger.error(f"No trained model found for {symbol}")
                return None

            # Load the model
            with open(model_path, "rb") as _:
                # model_data = pickle.load(f)
                pass

            # Use Analyst's prediction functionality
            if self.analyst:
                prediction = await self.analyst.run_analysis_pipeline()
                return prediction

            return None

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            return None

    @handle_errors(
        exceptions=(Exception,), default_return=[], context="get_training_status"
    )
    async def get_training_status(self, symbol: str) -> List[Dict[str, Any]]:
        """Get training status and history for a symbol."""
        try:
            # Get training sessions from database
            training_sessions = await self.db_manager.get_collection(
                "training_sessions"
            )

            # Filter by symbol
            symbol_sessions = [
                session
                for session in training_sessions
                if session.get("symbol") == symbol
            ]

            # Get model checkpoints
            model_checkpoints = await self.db_manager.get_collection(
                "model_checkpoints"
            )
            symbol_checkpoints = [
                checkpoint
                for checkpoint in model_checkpoints
                if checkpoint.get("symbol") == symbol
            ]

            # Combine and sort by date
            all_records = symbol_sessions + symbol_checkpoints
            all_records.sort(key=lambda x: x.get("trained_at", ""), reverse=True)

            return all_records

        except Exception as e:
            self.logger.error(f"Failed to get training status: {e}", exc_info=True)
            return []

    @handle_errors(
        exceptions=(Exception,), default_return=[], context="list_available_models"
    )
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available trained models."""
        try:
            models = []

            # Check models directory
            for file in os.listdir(self.models_dir):
                if file.endswith(".pkl"):
                    model_info = {
                        "filename": file,
                        "path": os.path.join(self.models_dir, file),
                        "size": os.path.getsize(os.path.join(self.models_dir, file)),
                        "modified": datetime.fromtimestamp(
                            os.path.getmtime(os.path.join(self.models_dir, file))
                        ).isoformat(),
                    }
                    models.append(model_info)

            # Get model checkpoints from database
            model_checkpoints = await self.db_manager.get_collection(
                "model_checkpoints"
            )
            for checkpoint in model_checkpoints:
                models.append(
                    {
                        "symbol": checkpoint.get("symbol"),
                        "session_id": checkpoint.get("session_id"),
                        "trained_at": checkpoint.get("trained_at"),
                        "model_paths": checkpoint.get("model_paths", {}),
                    }
                )

            return models

        except Exception as e:
            self.logger.error(f"Failed to list models: {e}", exc_info=True)
            return []

    @handle_errors(
        exceptions=(Exception,), default_return=False, context="export_model"
    )
    async def export_model(self, symbol: str, export_path: str) -> bool:
        """Export a trained model to a specified path."""
        try:
            # Find the model file for the symbol
            model_files = [
                f
                for f in os.listdir(self.models_dir)
                if f.startswith(symbol) and f.endswith(".pkl")
            ]

            if not model_files:
                self.logger.error(f"No model found for {symbol}")
                return False

            # Use the most recent model
            latest_model = sorted(model_files)[-1]
            source_path = os.path.join(self.models_dir, latest_model)

            # Copy the model file
            import shutil

            shutil.copy2(source_path, export_path)

            self.logger.info(
                f"Model exported successfully: {source_path} -> {export_path}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Model export failed: {e}", exc_info=True)
            return False
