#!/usr/bin/env python3
"""Step 3: Bayesian Parameter Optimization for HMM Regime Discovery using Optuna.

This module replaces the grid search optimization with Bayesian optimization
using Optuna for more efficient and effective parameter tuning.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.domain import (
    comprehensive_data_validation,
    ensure_data_integrity,
    monitor_feature_engineering,
    monitor_step_execution,
    quality_gate,
    secure_data_processing,
    secure_step_execution,
    validate_pipeline_step
)
from src.core.decorators import handles_errors, validates
from src.utils.logger import system_logger

logger = system_logger.getChild("Step3BayesianParameterOptimization")

# Try to import Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("⚠️ Optuna not available, falling back to grid search")

# Try to import HMM components
try:
    from hmmlearn import hmm
    from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("⚠️ HMM components not available")


class BayesianParameterOptimizationStep:
    """Step 3: Bayesian Parameter Optimization for HMM Regime Discovery."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("BayesianParameterOptimizationStep")
        self.start_time = None
        self.optimization_results = {}
        self.best_params = {}
        self._initialize_components()

    @secure_step_execution
    def _initialize_components(self) -> None:
        """Initialize Bayesian parameter optimization components."""
        self.logger.info("🔧 Initializing Bayesian parameter optimization components...")
        
        if not OPTUNA_AVAILABLE:
            self.logger.error("❌ Optuna not available - cannot perform Bayesian optimization")
            raise ImportError("Optuna is required for Bayesian parameter optimization")
        
        if not HMM_AVAILABLE:
            self.logger.error("❌ HMM components not available")
            raise ImportError("HMM components are required for regime discovery")
        
        # Initialize Optuna study
        self.study_name = f"hmm_regime_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.sampler = TPESampler(seed=42)
        self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        self.logger.info("✅ Bayesian parameter optimization components initialized successfully")

    @handles_errors(fallback=False)
    @secure_step_execution
    async def initialize(self) -> bool:
        """Initialize the Bayesian parameter optimization step."""
        try:
            self.logger.info("🚀 Initializing Bayesian parameter optimization step...")
            
            # Load optimization configuration
            optimization_config = self.config.get("bayesian_optimization", {})
            self.n_trials = optimization_config.get("n_trials", 100)
            self.timeout_minutes = optimization_config.get("timeout_minutes", 30)
            self.cv_folds = optimization_config.get("cv_folds", 3)
            self.random_state = optimization_config.get("random_state", 42)
            
            self.logger.info(f"📋 Optimization configuration:")
            self.logger.info(f"   - Trials: {self.n_trials}")
            self.logger.info(f"   - Timeout: {self.timeout_minutes} minutes")
            self.logger.info(f"   - CV folds: {self.cv_folds}")
            self.logger.info(f"   - Random state: {self.random_state}")
            
            self.logger.info("✅ Bayesian parameter optimization step initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Bayesian parameter optimization step: {e}")
            return False

    @monitor_step_execution
    @secure_step_execution
    @validates()
    @handles_errors(fallback=False)
    async def execute(self) -> bool:
        """Execute the Bayesian parameter optimization step."""
        try:
            self.logger.info("🎯 Starting Bayesian parameter optimization for HMM regime discovery...")
            self.start_time = time.time()
            
            # Step 1: Load and validate data
            data_loaded = await self._load_and_validate_data()
            if not data_loaded.get("success", False):
                self.logger.error("Failed to load and validate data")
                return False
            
            # Step 2: Create Optuna study
            study = optuna.create_study(
                direction="maximize",
                sampler=self.sampler,
                pruner=self.pruner,
                study_name=self.study_name
            )
            
            # Step 3: Define optimization objective
            def objective(trial):
                return self._optimization_objective(trial, data_loaded["data"], data_loaded["features"])
            
            # Step 4: Run optimization
            self.logger.info(f"🔍 Starting optimization with {self.n_trials} trials...")
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout_minutes * 60,
                show_progress_bar=True
            )
            
            # Step 5: Extract best parameters
            self.best_params = study.best_params
            self.optimization_results = {
                "best_params": self.best_params,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
                "study_name": self.study_name,
                "optimization_time": time.time() - self.start_time
            }
            
            # Step 6: Validate best parameters
            validation_result = await self._validate_best_parameters(data_loaded["data"], data_loaded["features"])
            self.optimization_results["validation"] = validation_result
            
            # Step 7: Save optimization results
            await self._save_optimization_results()
            
            # Step 8: Generate optimization reports
            await self._generate_optimization_reports(study)
            
            execution_time = time.time() - self.start_time
            self.logger.info(f"✅ Bayesian parameter optimization completed successfully in {execution_time:.2f}s")
            self.logger.info(f"📊 Best score: {study.best_value:.4f}")
            self.logger.info(f"🎯 Best parameters: {self.best_params}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute Bayesian parameter optimization: {e}")
            return False

    def _optimization_objective(self, trial, data: pd.DataFrame, features: pd.DataFrame) -> float:
        """Define the optimization objective for Optuna."""
        try:
            # Suggest HMM parameters
            n_components = trial.suggest_int("n_components", 2, 8)
            covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])
            n_iter = trial.suggest_int("n_iter", 50, 200)
            tol = trial.suggest_float("tol", 1e-6, 1e-2, log=True)
            reg_covar = trial.suggest_float("reg_covar", 1e-7, 1e-2, log=True)
            
            # Suggest clustering parameters
            n_clusters = trial.suggest_int("n_clusters", 10, 30)
            clustering_method = trial.suggest_categorical("clustering_method", ["kmeans", "gaussian_mixture", "spectral"])
            
            # Suggest feature selection parameters
            feature_selection_method = trial.suggest_categorical("feature_selection_method", ["variance", "correlation", "mutual_info"])
            max_features = trial.suggest_int("max_features", 20, min(100, features.shape[1]))
            
            # Prepare features
            features_processed = self._process_features_for_trial(features, feature_selection_method, max_features)
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_processed)
            
            # Train HMM
            hmm_model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                reg_covar=reg_covar,
                random_state=42
            )
            
            # Use subset for faster training
            max_samples = min(50000, features_scaled.shape[0])
            if features_scaled.shape[0] > max_samples:
                indices = np.random.choice(features_scaled.shape[0], max_samples, replace=False)
                features_subset = features_scaled[indices]
            else:
                features_subset = features_scaled
            
            hmm_model.fit(features_subset)
            
            # Get HMM states
            hmm_states = hmm_model.predict(features_scaled)
            hmm_probs = hmm_model.predict_proba(features_scaled)
            
            # Create composite features
            composite_features = self._create_composite_features(features_processed, hmm_states, hmm_probs)
            composite_scaled = scaler.fit_transform(composite_features)
            
            # Perform clustering
            if clustering_method == "kmeans":
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            elif clustering_method == "gaussian_mixture":
                clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
            else:  # spectral
                clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42)
            
            cluster_labels = clusterer.fit_predict(composite_scaled)
            
            # Calculate objective score
            score = self._calculate_objective_score(features_scaled, hmm_states, cluster_labels, hmm_model)
            
            # Report intermediate value for pruning
            trial.report(score, 0)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return -float('inf')

    def _process_features_for_trial(self, features: pd.DataFrame, method: str, max_features: int) -> pd.DataFrame:
        """Process features for a specific trial."""
        try:
            if method == "variance":
                # Select features with highest variance
                variances = features.var()
                selected_features = variances.nlargest(max_features).index
            elif method == "correlation":
                # Remove highly correlated features
                corr_matrix = features.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
                features_reduced = features.drop(columns=to_drop)
                selected_features = features_reduced.columns[:max_features]
            else:  # mutual_info
                # Select features with highest mutual information with target (simplified)
                selected_features = features.columns[:max_features]
            
            return features[selected_features]
            
        except Exception as e:
            self.logger.warning(f"Feature processing failed: {e}")
            return features.iloc[:, :max_features]

    def _create_composite_features(self, features: pd.DataFrame, hmm_states: np.ndarray, hmm_probs: np.ndarray) -> pd.DataFrame:
        """Create composite features combining HMM states with original features."""
        try:
            composite_df = features.copy()
            composite_df['hmm_state'] = hmm_states
            composite_df['hmm_state_prob_max'] = np.max(hmm_probs, axis=1)
            composite_df['hmm_state_entropy'] = -np.sum(hmm_probs * np.log(hmm_probs + 1e-10), axis=1)
            
            # Add HMM state probabilities
            for i in range(hmm_probs.shape[1]):
                composite_df[f'hmm_state_prob_{i}'] = hmm_probs[:, i]
            
            # Add feature interactions with HMM states
            key_features = ['price_momentum_10', 'volatility_20', 'volume_ratio_10', 'rsi', 'adx']
            for feature in key_features:
                if feature in composite_df.columns:
                    composite_df[f'{feature}_x_hmm_state'] = composite_df[feature] * composite_df['hmm_state']
                    composite_df[f'{feature}_x_hmm_entropy'] = composite_df[feature] * composite_df['hmm_state_entropy']
            
            return composite_df
            
        except Exception as e:
            self.logger.warning(f"Composite feature creation failed: {e}")
            return features

    def _calculate_objective_score(self, features: np.ndarray, hmm_states: np.ndarray, cluster_labels: np.ndarray, hmm_model) -> float:
        """Calculate the objective score for optimization."""
        try:
            # HMM score (log-likelihood)
            hmm_score = hmm_model.score(features)
            
            # Clustering quality metrics
            try:
                silhouette = silhouette_score(features, cluster_labels)
            except:
                silhouette = 0.0
            
            try:
                calinski_harabasz = calinski_harabasz_score(features, cluster_labels)
            except:
                calinski_harabasz = 0.0
            
            try:
                davies_bouldin = davies_bouldin_score(features, cluster_labels)
                davies_bouldin_score_normalized = 1.0 / (1.0 + davies_bouldin)  # Invert to make higher better
            except:
                davies_bouldin_score_normalized = 0.0
            
            # Regime stability (inverse of regime switching frequency)
            regime_changes = np.sum(np.diff(hmm_states) != 0)
            regime_stability = 1.0 / (1.0 + regime_changes / len(hmm_states))
            
            # Regime balance (inverse of regime size variance)
            unique_states, counts = np.unique(hmm_states, return_counts=True)
            regime_balance = 1.0 / (1.0 + np.std(counts) / np.mean(counts))
            
            # Combined score with weights
            weights = {
                'hmm_score': 0.3,
                'silhouette': 0.25,
                'calinski_harabasz': 0.15,
                'davies_bouldin': 0.15,
                'regime_stability': 0.1,
                'regime_balance': 0.05
            }
            
            # Normalize scores to [0, 1] range
            hmm_score_normalized = max(0, min(1, (hmm_score + 1000) / 1000))  # Rough normalization
            calinski_harabasz_normalized = max(0, min(1, calinski_harabasz / 1000))  # Rough normalization
            
            combined_score = (
                weights['hmm_score'] * hmm_score_normalized +
                weights['silhouette'] * max(0, silhouette) +
                weights['calinski_harabasz'] * calinski_harabasz_normalized +
                weights['davies_bouldin'] * davies_bouldin_score_normalized +
                weights['regime_stability'] * regime_stability +
                weights['regime_balance'] * regime_balance
            )
            
            return combined_score
            
        except Exception as e:
            self.logger.warning(f"Score calculation failed: {e}")
            return 0.0

    @handles_errors(
        default_return={"success": False, "error": "Data loading failed"},
        context="load_and_validate_data"
    )
    @validates()
    @ensure_data_integrity
    async def _load_and_validate_data(self) -> dict[str, Any]:
        """Load and validate data for parameter optimization."""
        try:
            self.logger.info("📊 Loading and validating data for Bayesian parameter optimization...")
            
            # Get data parameters from config
            symbol = self.config.get("SYMBOL", "ETHUSDT")
            exchange = self.config.get("EXCHANGE", "BINANCE")
            timeframe = self.config.get("TIMEFRAME", "1m")
            data_dir = self.config.get("DATA_DIR", "data_cache")
            
            # Load klines data
            klines_path = Path(data_dir) / f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
            
            if not klines_path.exists():
                self.logger.error(f"❌ Klines file not found: {klines_path}")
                return {
                    "success": False,
                    "error": f"Klines file not found: {klines_path}"
                }
            
            # Load data
            df = pd.read_parquet(klines_path)
            
            if df.empty:
                self.logger.error("❌ Data is empty")
                return {
                    "success": False,
                    "error": "Data is empty"
                }
            
            # Prepare features for optimization
            features = await self._prepare_features_for_optimization(df)
            
            self.logger.info(f"✅ Data loaded and validated: {len(df):,} rows, {len(features.columns)} features")
            
            return {
                "success": True,
                "data": df,
                "features": features,
                "data_info": {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "date_range": {
                        "start": df["timestamp"].min().isoformat(),
                        "end": df["timestamp"].max().isoformat()
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load and validate data: {e}")
            return {"success": False, "error": str(e)}

    @handles_errors(fallback=pd.DataFrame())
    @monitor_feature_engineering()
    @validates()
    async def _prepare_features_for_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for parameter optimization."""
        try:
            self.logger.info("🔧 Preparing features for Bayesian parameter optimization...")
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Create comprehensive feature set
            features = pd.DataFrame()
            features['timestamp'] = df['timestamp']
            
            # Price-based features
            features['price_momentum_5'] = df['close'].pct_change(5)
            features['price_momentum_10'] = df['close'].pct_change(10)
            features['price_momentum_20'] = df['close'].pct_change(20)
            
            # Volume features
            features['volume_momentum_5'] = df['volume'].pct_change(5)
            features['volume_momentum_10'] = df['volume'].pct_change(10)
            features['volume_ratio_5'] = df['volume'] / df['volume'].rolling(window=5).mean()
            features['volume_ratio_10'] = df['volume'] / df['volume'].rolling(window=10).mean()
            features['volume_ratio_20'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            # Volatility features
            features['volatility_5'] = df['close'].pct_change().rolling(window=5).std()
            features['volatility_10'] = df['close'].pct_change().rolling(window=10).std()
            features['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
            features['ewma_volatility_20'] = df['close'].pct_change().ewm(span=20).std()
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(df['close'])
            features['macd'] = self._calculate_macd(df['close'])
            features['atr'] = self._calculate_atr(df)
            features['adx'] = self._calculate_adx(df)
            
            # Bollinger Bands
            bb_features = self._calculate_bollinger_bands(df['close'])
            features = pd.concat([features, bb_features], axis=1)
            
            # Moving averages
            features['sma_20'] = df['close'].rolling(window=20).mean()
            features['sma_50'] = df['close'].rolling(window=50).mean()
            features['ema_12'] = df['close'].ewm(span=12).mean()
            features['ema_26'] = df['close'].ewm(span=26).mean()
            
            # Price position relative to MAs
            features['price_vs_sma20'] = (df['close'] - features['sma_20']) / features['sma_20']
            features['price_vs_sma50'] = (df['close'] - features['sma_50']) / features['sma_50']
            
            # Feature interactions
            features['momentum_volume_interaction'] = features['price_momentum_10'] * features['volume_ratio_10']
            features['volatility_volume_interaction'] = features['volatility_20'] * features['volume_ratio_20']
            features['rsi_momentum_interaction'] = features['rsi'] * features['price_momentum_10']
            
            # Clean features
            hmm_features = features.drop('timestamp', axis=1)
            hmm_features = hmm_features.fillna(0)
            
            self.logger.info(f"✅ Features prepared: {len(hmm_features.columns)} features, {len(hmm_features)} samples")
            
            return hmm_features
            
        except Exception as e:
            self.logger.error(f"Failed to prepare features: {e}")
            return pd.DataFrame()

    # Technical indicator calculation methods (same as in original step3)
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        tr_smooth = tr.rolling(window=window).mean()
        dm_plus_smooth = dm_plus.rolling(window=window).mean()
        dm_minus_smooth = dm_minus.rolling(window=window).mean()
        
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=window).mean()
        
        return adx

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        bb_upper = sma + std * num_std
        bb_lower = sma - std * num_std
        bb_width = (bb_upper - bb_lower) / sma
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        
        bb_features = pd.DataFrame({
            'bb_upper': bb_upper,
            'bb_middle': sma,
            'bb_lower': bb_lower,
            'bb_width': bb_width,
            'bb_position': bb_position
        })
        
        return bb_features

    @handles_errors(fallback=False)
    async def _validate_best_parameters(self, data: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
        """Validate the best parameters found by optimization."""
        try:
            self.logger.info("🔍 Validating best parameters...")
            
            # Use best parameters to train final model
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train HMM with best parameters
            hmm_model = hmm.GaussianHMM(
                n_components=self.best_params['n_components'],
                covariance_type=self.best_params['covariance_type'],
                n_iter=self.best_params['n_iter'],
                tol=self.best_params['tol'],
                reg_covar=self.best_params['reg_covar'],
                random_state=42
            )
            
            hmm_model.fit(features_scaled)
            hmm_states = hmm_model.predict(features_scaled)
            hmm_probs = hmm_model.predict_proba(features_scaled)
            
            # Create composite features and cluster
            composite_features = self._create_composite_features(features, hmm_states, hmm_probs)
            composite_scaled = scaler.fit_transform(composite_features)
            
            # Train clustering model
            if self.best_params['clustering_method'] == "kmeans":
                clusterer = KMeans(n_clusters=self.best_params['n_clusters'], random_state=42, n_init=10)
            elif self.best_params['clustering_method'] == "gaussian_mixture":
                clusterer = GaussianMixture(n_components=self.best_params['n_clusters'], random_state=42)
            else:
                clusterer = SpectralClustering(n_clusters=self.best_params['n_clusters'], random_state=42)
            
            cluster_labels = clusterer.fit_predict(composite_scaled)
            
            # Calculate validation metrics
            validation_metrics = {
                'hmm_score': hmm_model.score(features_scaled),
                'silhouette_score': silhouette_score(composite_scaled, cluster_labels),
                'calinski_harabasz_score': calinski_harabasz_score(composite_scaled, cluster_labels),
                'davies_bouldin_score': davies_bouldin_score(composite_scaled, cluster_labels),
                'n_regimes': len(np.unique(hmm_states)),
                'n_clusters': len(np.unique(cluster_labels)),
                'regime_stability': 1.0 / (1.0 + np.sum(np.diff(hmm_states) != 0) / len(hmm_states))
            }
            
            self.logger.info("✅ Best parameters validation completed")
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to validate best parameters: {e}")
            return {"error": str(e)}

    @handles_errors(fallback=False)
    async def _save_optimization_results(self) -> None:
        """Save optimization results to file."""
        try:
            self.logger.info("💾 Saving optimization results...")
            
            # Create output directory
            output_dir = Path("data/optimization_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_dir / f"bayesian_optimization_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(self.optimization_results, f, indent=2, default=str)
            
            self.logger.info(f"✅ Optimization results saved to: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save optimization results: {e}")

    @handles_errors(fallback=False)
    async def _generate_optimization_reports(self, study) -> None:
        """Generate optimization reports."""
        try:
            self.logger.info("📊 Generating optimization reports...")
            
            # Create reports directory
            reports_dir = Path("data/optimization_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate parameter importance report
            try:
                importance = optuna.importance.get_param_importances(study)
                importance_file = reports_dir / f"parameter_importance_{timestamp}.json"
                with open(importance_file, 'w') as f:
                    json.dump(importance, f, indent=2)
                self.logger.info(f"✅ Parameter importance saved to: {importance_file}")
            except Exception as e:
                self.logger.warning(f"Could not generate parameter importance: {e}")
            
            # Generate optimization history
            history = []
            for trial in study.trials:
                history.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                })
            
            history_file = reports_dir / f"optimization_history_{timestamp}.json"
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            self.logger.info(f"✅ Optimization history saved to: {history_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization reports: {e}")


@monitor_step_execution
@secure_step_execution
@validates()
@handles_errors(fallback=False)
async def run_bayesian_optimization(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_dir: str = None,
    force_rerun: bool = False,
    **kwargs: Any
) -> bool:
    """Run the Bayesian parameter optimization step.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        exchange: Exchange name (e.g., "BINANCE")
        timeframe: Timeframe (e.g., "1m")
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force re-run even if results exist
        **kwargs: Additional arguments

    Returns:
        bool: True if successful, False otherwise
    """
    start_time = time.time()
    
    try:
        logger = system_logger.getChild('Step3BayesianParameterOptimization')
        
        if data_dir is None:
            data_dir = "data_cache"
        
        logger.info('=' * 80)
        logger.info('🚀 STEP 3: Bayesian Parameter Optimization for HMM Regime Discovery')
        logger.info('=' * 80)
        logger.info(f'🎯 Symbol: {symbol}')
        logger.info(f'🏢 Exchange: {exchange}')
        logger.info(f'📊 Timeframe: {timeframe}')
        logger.info(f'📁 Data directory: {data_dir}')
        logger.info(f'🔄 Force rerun: {force_rerun}')
        logger.info(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info('=' * 80)
        
        # Create configuration
        config = {
            'SYMBOL': symbol,
            'EXCHANGE': exchange,
            'TIMEFRAME': timeframe,
            'DATA_DIR': data_dir,
            'bayesian_optimization': {
                'n_trials': kwargs.get('n_trials', 100),
                'timeout_minutes': kwargs.get('timeout_minutes', 30),
                'cv_folds': kwargs.get('cv_folds', 3),
                'random_state': kwargs.get('random_state', 42)
            }
        }
        
        # Initialize and run optimization
        logger.info('🔧 Initializing Bayesian parameter optimization step...')
        step = BayesianParameterOptimizationStep(config)
        
        initialized = await step.initialize()
        if not initialized:
            logger.error('❌ Failed to initialize Bayesian parameter optimization step')
            return False
        
        logger.info('🎯 Executing Bayesian parameter optimization...')
        success = await step.execute()
        
        if success:
            logger.info('✅ Step 3: Bayesian Parameter Optimization completed successfully')
            logger.info(f'📊 Best parameters: {step.best_params}')
            logger.info(f'📈 Best score: {step.optimization_results.get("best_value", "N/A")}')
            
            total_elapsed = time.time() - start_time
            logger.info('=' * 80)
            logger.info('🎉 STEP 3 EXECUTION SUMMARY')
            logger.info('=' * 80)
            logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
            logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info('✅ SUCCESS')
            logger.info('=' * 80)
            
            return True
        else:
            logger.error('❌ Step 3: Bayesian Parameter Optimization failed')
            
            total_elapsed = time.time() - start_time
            logger.info('=' * 80)
            logger.info('💥 STEP 3 EXECUTION SUMMARY')
            logger.info('=' * 80)
            logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
            logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info('❌ FAILED')
            logger.info('=' * 80)
            
            return False
            
    except Exception as e:
        logger.exception(f'❌ Step 3: Bayesian Parameter Optimization failed with exception: {e}')
        
        total_elapsed = time.time() - start_time
        logger.info('=' * 80)
        logger.info('💥 STEP 3 EXECUTION SUMMARY')
        logger.info('=' * 80)
        logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
        logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info('❌ FAILED')
        logger.info(f'   Exception: {e}')
        logger.info('=' * 80)
        
        return False


if __name__ == "__main__":
    # Example usage
    success = asyncio.run(run_bayesian_optimization(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        n_trials=50,
        timeout_minutes=15
    ))
    
    if success:
        print("✅ Bayesian parameter optimization completed successfully!")
    else:
        print("❌ Bayesian parameter optimization failed")