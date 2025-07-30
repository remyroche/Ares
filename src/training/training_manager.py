# src/training/training_manager.py

import asyncio
import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import system_logger
from src.config import CONFIG, settings
from src.database.sqlite_manager import SQLiteManager
from src.exchange.binance import exchange
from src.utils.error_handler import handle_errors, handle_data_processing_errors
from src.analyst.analyst import Analyst
from src.supervisor.main import Supervisor
from src.utils.state_manager import StateManager

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
    """
    
    def __init__(self, db_manager: SQLiteManager):
        self.db_manager = db_manager
        self.logger = system_logger.getChild('TrainingManager')
        self.models_dir = "models"
        self.data_dir = "data/training"
        self.reports_dir = "reports"
        
        # Create necessary directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Initialize components
        self.state_manager = StateManager()
        self.analyst = None
        self.supervisor = None
        
        # Training state
        self.current_training_session = None
        self.training_history = {}
        
    async def initialize_components(self, symbol: str):
        """Initialize Analyst and Supervisor components for training."""
        try:
            # Initialize exchange client
            from src.exchange.binance import exchange
            
            # Initialize Analyst
            self.analyst = Analyst(exchange, self.state_manager)
            
            # Initialize Supervisor
            self.supervisor = Supervisor(
                exchange_client=exchange,
                state_manager=self.state_manager,
                db_manager=self.db_manager
            )
            
            self.logger.info(f"Components initialized for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}", exc_info=True)
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="full_training_pipeline"
    )
    async def run_full_training(self, symbol: str, exchange_name: str = "BINANCE") -> bool:
        """
        Run complete training pipeline for a specific token.
        
        This orchestrates the entire training process by leveraging existing components:
        1. Data collection and preparation (Analyst)
        2. Feature engineering (Analyst)
        3. Model training (Analyst + Predictive Ensembles)
        4. Hyperparameter optimization (Supervisor)
        5. Walk-forward validation (Backtesting)
        6. Monte Carlo validation (Backtesting)
        7. A/B testing setup (Supervisor)
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            exchange_name: Exchange name (e.g., "BINANCE")
            
        Returns:
            bool: True if training successful, False otherwise
        """
        self.logger.info(f"Starting full training pipeline for {symbol} on {exchange_name}")
        
        # Initialize training session
        session_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_training_session = {
            'session_id': session_id,
            'symbol': symbol,
            'exchange': exchange_name,
            'start_time': datetime.now().isoformat(),
            'status': 'running'
        }
        
        try:
            # Initialize components
            if not await self.initialize_components(symbol):
                return False
            
            # Step 1: Data Collection and Preparation
            self.logger.info("Step 1: Collecting and preparing historical data...")
            data_success = await self._collect_and_prepare_data(symbol, exchange_name)
            if not data_success:
                self.logger.error("Data collection failed. Aborting training.")
                return False
            
            # Step 2: Feature Engineering and Model Training
            self.logger.info("Step 2: Engineering features and training models...")
            model_success = await self._train_analyst_models(symbol)
            if not model_success:
                self.logger.error("Model training failed. Aborting training.")
                return False
            
            # Step 3: Hyperparameter Optimization
            self.logger.info("Step 3: Running hyperparameter optimization...")
            optimization_success = await self._run_hyperparameter_optimization(symbol)
            if not optimization_success:
                self.logger.error("Hyperparameter optimization failed. Aborting training.")
                return False
            
            # Step 4: Walk-Forward Validation
            self.logger.info("Step 4: Performing walk-forward validation...")
            walk_forward_success = await self._run_walk_forward_validation(symbol)
            if not walk_forward_success:
                self.logger.error("Walk-forward validation failed. Aborting training.")
                return False
            
            # Step 5: Monte Carlo Validation
            self.logger.info("Step 5: Performing Monte Carlo validation...")
            monte_carlo_success = await self._run_monte_carlo_validation(symbol)
            if not monte_carlo_success:
                self.logger.error("Monte Carlo validation failed. Aborting training.")
                return False
            
            # Step 6: A/B Testing Setup
            self.logger.info("Step 6: Setting up A/B testing...")
            ab_test_success = await self._setup_ab_testing(symbol)
            if not ab_test_success:
                self.logger.error("A/B testing setup failed. Aborting training.")
                return False
            
            # Step 7: Save Training Results
            self.logger.info("Step 7: Saving training results...")
            await self._save_training_results(symbol, session_id)
            
            self.current_training_session['status'] = 'completed'
            self.current_training_session['end_time'] = datetime.now().isoformat()
            
            self.logger.info(f"Full training pipeline completed successfully for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Full training pipeline failed: {e}", exc_info=True)
            self.current_training_session['status'] = 'failed'
            self.current_training_session['error'] = str(e)
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="data_collection"
    )
    async def _collect_and_prepare_data(self, symbol: str, exchange_name: str) -> bool:
        """Collect and prepare historical data for training using existing Analyst functionality."""
        try:
            self.logger.info(f"Collecting historical data for {symbol}...")
            
            # Use Analyst's data loading functionality
            lookback_days = CONFIG['MODEL_TRAINING']['data_retention_days']
            end_time_ms = exchange._get_timestamp()
            start_time_ms = end_time_ms - int(timedelta(days=lookback_days).total_seconds() * 1000)
            
            # Fetch klines data
            klines_raw = await exchange.get_klines(symbol, "1h", limit=5000)
            if not klines_raw:
                self.logger.error(f"Failed to fetch klines data for {symbol}")
                return False
            
            # Convert to DataFrame using Analyst's formatting
            klines_df = self.analyst._format_klines_data(klines_raw)
            
            # Fetch aggregated trades
            agg_trades_raw = await exchange.get_historical_agg_trades(symbol, start_time_ms, end_time_ms)
            agg_trades_df = pd.DataFrame(agg_trades_raw) if agg_trades_raw else pd.DataFrame()
            
            # Fetch futures data
            futures_data = await exchange.get_historical_futures_data(symbol, start_time_ms, end_time_ms)
            futures_df = pd.DataFrame(futures_data.get('funding_rates', [])) if futures_data else pd.DataFrame()
            
            # Save raw data
            data_file = os.path.join(self.data_dir, f"{symbol}_historical_data.pkl")
            with open(data_file, 'wb') as f:
                pickle.dump({
                    'klines': klines_df,
                    'agg_trades': agg_trades_df,
                    'futures': futures_df,
                    'symbol': symbol,
                    'collected_at': datetime.now().isoformat()
                }, f)
            
            self.logger.info(f"Collected {len(klines_df)} klines, {len(agg_trades_df)} trades, {len(futures_df)} futures records for {symbol}")
            return len(klines_df) >= CONFIG['MODEL_TRAINING']['min_data_points']
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}", exc_info=True)
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="model_training"
    )
    async def _train_analyst_models(self, symbol: str) -> bool:
        """Train Analyst models using existing functionality."""
        try:
            self.logger.info(f"Training Analyst models for {symbol}...")
            
            # Load collected data
            data_file = os.path.join(self.data_dir, f"{symbol}_historical_data.pkl")
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            # Use Analyst's data preparation and model training
            success = await self.analyst.load_and_prepare_historical_data(
                historical_klines=data['klines'],
                historical_agg_trades=data['agg_trades'],
                historical_futures=data['futures'],
                fold_id=f"training_{symbol}"
            )
            
            if success:
                self.logger.info(f"Analyst models trained successfully for {symbol}")
                return True
            else:
                self.logger.error(f"Analyst model training failed for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Model training failed: {e}", exc_info=True)
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="hyperparameter_optimization"
    )
    async def _run_hyperparameter_optimization(self, symbol: str) -> bool:
        """Run hyperparameter optimization using Supervisor functionality."""
        try:
            self.logger.info(f"Running hyperparameter optimization for {symbol}...")
            
            # Use Supervisor's optimization functionality
            optimization_result = await self.supervisor.optimizer.implement_global_system_optimization(
                historical_pnl_data=pd.DataFrame(),  # Will be loaded from database
                strategy_breakdown_data={},  # Will be loaded from database
                checkpoint_file_path=os.path.join(self.models_dir, f"{symbol}_optimization_checkpoint.pkl")
            )
            
            if optimization_result:
                self.logger.info(f"Hyperparameter optimization completed for {symbol}")
                return True
            else:
                self.logger.error(f"Hyperparameter optimization failed for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {e}", exc_info=True)
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="walk_forward_validation"
    )
    async def _run_walk_forward_validation(self, symbol: str) -> bool:
        """Run walk-forward validation using backtesting functionality."""
        try:
            self.logger.info(f"Running walk-forward validation for {symbol}...")
            
            # Import backtesting functionality
            from backtesting.ares_deep_analyzer import run_walk_forward_analysis
            from backtesting.ares_data_preparer import calculate_and_label_regimes, get_sr_levels
            
            # Load data
            data_file = os.path.join(self.data_dir, f"{symbol}_historical_data.pkl")
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            # Prepare data for backtesting
            daily_df = data['klines'].resample('D').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 
                'close': 'last', 'volume': 'sum'
            }).dropna()
            
            sr_levels = get_sr_levels(daily_df)
            
            # Use current best parameters
            best_params = CONFIG.get('BEST_PARAMS', {})
            
            # Prepare data with current parameters
            prepared_data = calculate_and_label_regimes(
                data['klines'], data['agg_trades'], data['futures'], 
                best_params, sr_levels
            )
            
            # Run walk-forward analysis
            wfa_report = run_walk_forward_analysis(prepared_data, best_params)
            
            # Save walk-forward results
            wfa_file = os.path.join(self.reports_dir, f"{symbol}_walk_forward_report.txt")
            with open(wfa_file, 'w') as f:
                f.write(wfa_report)
            
            self.logger.info(f"Walk-forward validation completed for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Walk-forward validation failed: {e}", exc_info=True)
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="monte_carlo_validation"
    )
    async def _run_monte_carlo_validation(self, symbol: str) -> bool:
        """Run Monte Carlo validation using backtesting functionality."""
        try:
            self.logger.info(f"Running Monte Carlo validation for {symbol}...")
            
            # Import backtesting functionality
            from backtesting.ares_deep_analyzer import run_monte_carlo_simulation
            from backtesting.ares_data_preparer import calculate_and_label_regimes, get_sr_levels
            
            # Load data
            data_file = os.path.join(self.data_dir, f"{symbol}_historical_data.pkl")
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            # Prepare data for backtesting
            daily_df = data['klines'].resample('D').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 
                'close': 'last', 'volume': 'sum'
            }).dropna()
            
            sr_levels = get_sr_levels(daily_df)
            
            # Use current best parameters
            best_params = CONFIG.get('BEST_PARAMS', {})
            
            # Prepare data with current parameters
            prepared_data = calculate_and_label_regimes(
                data['klines'], data['agg_trades'], data['futures'], 
                best_params, sr_levels
            )
            
            # Run Monte Carlo simulation
            mc_curves, base_portfolio, mc_report = run_monte_carlo_simulation(prepared_data, best_params)
            
            # Save Monte Carlo results
            mc_file = os.path.join(self.reports_dir, f"{symbol}_monte_carlo_report.txt")
            with open(mc_file, 'w') as f:
                f.write(mc_report)
            
            self.logger.info(f"Monte Carlo validation completed for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Monte Carlo validation failed: {e}", exc_info=True)
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="ab_testing_setup"
    )
    async def _setup_ab_testing(self, symbol: str) -> bool:
        """Set up A/B testing using Supervisor functionality."""
        try:
            self.logger.info(f"Setting up A/B testing for {symbol}...")
            
            # Create A/B test configuration
            ab_duration = CONFIG['MODEL_TRAINING']['ab_test_duration_days']
            ab_config = {
                'symbol': symbol,
                'start_date': datetime.now().isoformat(),
                'end_date': (datetime.now() + timedelta(days=ab_duration)).isoformat(),
                'duration_days': ab_duration,
                'status': 'active',
                'models': {
                    'model_a': 'current_model',
                    'model_b': 'new_model'
                },
                'metrics': ['accuracy', 'sharpe_ratio', 'max_drawdown']
            }
            
            # Save A/B test configuration to database
            await self.db_manager.set_document('ab_tests', f"{symbol}_ab_test", ab_config)
            
            self.logger.info("A/B testing setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"A/B testing setup failed: {e}", exc_info=True)
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="save_training_results"
    )
    async def _save_training_results(self, symbol: str, session_id: str) -> bool:
        """Save training results and metadata."""
        try:
            # Save training session metadata
            await self.db_manager.set_document('training_sessions', session_id, self.current_training_session)
            
            # Save model checkpoints
            model_checkpoint = {
                'symbol': symbol,
                'session_id': session_id,
                'trained_at': datetime.now().isoformat(),
                'model_paths': {
                    'analyst_models': os.path.join(CONFIG['CHECKPOINT_DIR'], "analyst_models"),
                    'supervisor_models': os.path.join(CONFIG['CHECKPOINT_DIR'], "supervisor_models"),
                    'optimization_results': os.path.join(self.models_dir, f"{symbol}_optimization_checkpoint.pkl")
                },
                'validation_reports': {
                    'walk_forward': os.path.join(self.reports_dir, f"{symbol}_walk_forward_report.txt"),
                    'monte_carlo': os.path.join(self.reports_dir, f"{symbol}_monte_carlo_report.txt")
                }
            }
            
            await self.db_manager.set_document('model_checkpoints', f"{symbol}_{session_id}", model_checkpoint)
            
            self.logger.info(f"Training results saved for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save training results: {e}", exc_info=True)
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="model_retraining"
    )
    async def retrain_models(self, symbol: str, exchange_name: str = "BINANCE") -> bool:
        """Retrain models with latest data."""
        self.logger.info(f"Starting model retraining for {symbol}")
        return await self.run_full_training(symbol, exchange_name)
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="model_import"
    )
    async def import_model(self, model_path: str, symbol: str) -> bool:
        """Import a trained model from file."""
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model data
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Validate model data structure
            if not isinstance(model_data, dict):
                self.logger.error("Invalid model file format")
                return False
            
            # Save to models directory with symbol prefix
            target_path = os.path.join(self.models_dir, f"{symbol}_imported_model.pkl")
            with open(target_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save import metadata to database
            import_metadata = {
                'symbol': symbol,
                'original_path': model_path,
                'imported_path': target_path,
                'imported_at': datetime.now().isoformat(),
                'model_info': model_data.get('model_info', {})
            }
            
            await self.db_manager.set_document('model_imports', f"{symbol}_import", import_metadata)
            
            self.logger.info(f"Model imported successfully for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model import failed: {e}", exc_info=True)
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="model_prediction"
    )
    async def predict(self, features: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        """Make predictions using the trained models for a symbol."""
        try:
            # Load the appropriate model for the symbol
            model_path = os.path.join(self.models_dir, f"{symbol}_best_model.pkl")
            if not os.path.exists(model_path):
                self.logger.error(f"No trained model found for {symbol}")
                return None
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Use Analyst's prediction functionality
            if self.analyst:
                prediction = await self.analyst.run_analysis_pipeline()
                return prediction
            
            return None
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            return None
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=[],
        context="get_training_status"
    )
    async def get_training_status(self, symbol: str) -> List[Dict[str, Any]]:
        """Get training status and history for a symbol."""
        try:
            # Get training sessions from database
            training_sessions = await self.db_manager.get_collection('training_sessions')
            
            # Filter by symbol
            symbol_sessions = [
                session for session in training_sessions 
                if session.get('symbol') == symbol
            ]
            
            # Get model checkpoints
            model_checkpoints = await self.db_manager.get_collection('model_checkpoints')
            symbol_checkpoints = [
                checkpoint for checkpoint in model_checkpoints
                if checkpoint.get('symbol') == symbol
            ]
            
            # Combine and sort by date
            all_records = symbol_sessions + symbol_checkpoints
            all_records.sort(key=lambda x: x.get('trained_at', ''), reverse=True)
            
            return all_records
            
        except Exception as e:
            self.logger.error(f"Failed to get training status: {e}", exc_info=True)
            return []
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=[],
        context="list_available_models"
    )
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available trained models."""
        try:
            models = []
            
            # Check models directory
            for file in os.listdir(self.models_dir):
                if file.endswith('.pkl'):
                    model_info = {
                        'filename': file,
                        'path': os.path.join(self.models_dir, file),
                        'size': os.path.getsize(os.path.join(self.models_dir, file)),
                        'modified': datetime.fromtimestamp(
                            os.path.getmtime(os.path.join(self.models_dir, file))
                        ).isoformat()
                    }
                    models.append(model_info)
            
            # Get model checkpoints from database
            model_checkpoints = await self.db_manager.get_collection('model_checkpoints')
            for checkpoint in model_checkpoints:
                models.append({
                    'symbol': checkpoint.get('symbol'),
                    'session_id': checkpoint.get('session_id'),
                    'trained_at': checkpoint.get('trained_at'),
                    'model_paths': checkpoint.get('model_paths', {})
                })
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}", exc_info=True)
            return []
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="export_model"
    )
    async def export_model(self, symbol: str, export_path: str) -> bool:
        """Export a trained model to a specified path."""
        try:
            # Find the model file for the symbol
            model_files = [
                f for f in os.listdir(self.models_dir) 
                if f.startswith(symbol) and f.endswith('.pkl')
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
            
            self.logger.info(f"Model exported successfully: {source_path} -> {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model export failed: {e}", exc_info=True)
            return False
        