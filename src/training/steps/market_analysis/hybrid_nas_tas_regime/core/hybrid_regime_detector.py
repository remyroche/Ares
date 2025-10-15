"""
Hybrid NAS-TAS Regime Detector

The core regime detection system that combines:
- Neural Architecture Search (NAS) from nas_regime/
- Tree Architecture Search (TAS) from ml_common TAS system
- Economic and financial relevance evaluation
- Advanced clustering algorithms

This replaces the HMM-based clustering system entirely.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import time
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import tprint for enhanced logging
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
        tprint_success, tprint_progress, tprint_performance, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError:
    def tprint(message: str, **kwargs) -> None:
        """Fallback tprint function if not available."""
        print(f"[HYBRID_NAS_TAS] {message}")
    def tprint_debug(message: str, **kwargs) -> None:
        print(f"[DEBUG] {message}")
    def tprint_info(message: str, **kwargs) -> None:
        print(f"[INFO] {message}")
    def tprint_warning(message: str, **kwargs) -> None:
        print(f"[WARNING] {message}")
    def tprint_error(message: str, **kwargs) -> None:
        print(f"[ERROR] {message}")
    def tprint_success(message: str, **kwargs) -> None:
        print(f"[SUCCESS] {message}")
    def tprint_progress(message: str, **kwargs) -> None:
        print(f"[PROGRESS] {message}")
    def tprint_timer(message: str, **kwargs) -> None:
        print(f"[TIMER] {message}")
    TPRINT_AVAILABLE = False

# Import sklearn components
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    tprint("⚠️ sklearn not available, clustering will use fallback methods", color="yellow")

from ..config.hybrid_regime_config import HybridRegimeConfig, RegimeCombinationStrategy
from ..components.tas_integration import TASIntegrationComponent
from ..components.nas_integration import NASIntegrationComponent
from ..evaluation.economic_evaluator import EconomicRegimeEvaluator
from .economic_clustering import EconomicClusterer
from .coherent_regime_modeling import CoherentRegimeModeler
from ..shared_utils.position_aware_trading import PositionAwareTradingAnalyzer, PositionAwareConfig

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None

logger = logging.getLogger(__name__)

@dataclass
class HybridRegimeResult:
    """Result from Hybrid NAS-TAS Regime Detection."""
    success: bool
    regime_predictions: np.ndarray
    regime_probabilities: np.ndarray
    economic_significance_scores: np.ndarray
    financial_relevance_scores: np.ndarray
    regime_stability_scores: np.ndarray
    transition_probabilities: np.ndarray
    combined_features: np.ndarray
    tas_contributions: Dict[str, Any]
    nas_contributions: Dict[str, Any]
    clustering_metrics: Dict[str, float]
    economic_clustering_metrics: Dict[str, Any]
    momentum_scores: np.ndarray
    volume_profiles: np.ndarray
    execution_time: float
    metadata: Dict[str, Any]
    micro_regime_predictions: Optional[np.ndarray] = None
    error_message: Optional[str] = None

class HybridNASTASRegimeDetector:
    """
    Hybrid NAS-TAS Regime Detector

    Combines Neural Architecture Search (NAS) and Tree Architecture Search (TAS)
    to create economically and financially relevant market regimes.
    """

    def __init__(self, config: HybridRegimeConfig):
        """Initialize the hybrid regime detector."""
        tprint("🎯 Initializing HybridNASTASRegimeDetector", color="blue")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        tprint(f"📊 Config: {config.n_regimes} regimes, strategy: {config.combination_strategy.value}", color="cyan")

        # Initialize component integrations
        tprint("🔧 Initializing TAS integration component", color="yellow")
        self.tas_integration = TASIntegrationComponent(config.tas_config)
        tprint("🧠 Initializing NAS integration component", color="yellow")
        self.nas_integration = NASIntegrationComponent(config.nas_config)
        tprint("📊 Initializing economic evaluator", color="yellow")
        self.economic_evaluator = EconomicRegimeEvaluator(config.economic_evaluation)
        tprint("🔍 Initializing economic clusterer", color="yellow")
        self.economic_clusterer = EconomicClusterer(config.clustering_config)
        tprint("📈 Initializing coherent regime modeler", color="yellow")
        self.regime_modeler = CoherentRegimeModeler(config.economic_evaluation)

        # Initialize position-aware analyzer for consistent win rate calculations
        tprint("⚖️ Initializing position-aware analyzer", color="yellow")
        self.position_analyzer = PositionAwareTradingAnalyzer(PositionAwareConfig())

        self.logger.info("✅ Hybrid NAS-TAS Regime Detector initialized")
        self.logger.info(f"   Combination Strategy: {config.combination_strategy.value}")
        self.logger.info(f"   TAS Weight: {config.tas_config.get('base_weight', 0.4)}")
        self.logger.info(f"   NAS Weight: {config.nas_config.get('base_weight', 0.6)}")
        self.logger.info(f"   Economic Evaluation: {config.economic_evaluation.get('enabled', True)}")
        self.logger.info(f"   Economic Clustering: {config.clustering_config.get('economic_clustering', True)}")
        self.logger.info(f"   Momentum Integration: {config.clustering_config.get('momentum_integration', True)}")
        self.logger.info(f"   Volume Integration: {config.clustering_config.get('volume_integration', True)}")
        self.logger.info(f"   Position-Aware Analysis: ✅ Enabled")
        
        tprint("✅ HybridNASTASRegimeDetector initialization complete", color="green")
        tprint(f"⚙️ TAS Weight: {config.tas_config.get('base_weight', 0.4)}, NAS Weight: {config.nas_config.get('base_weight', 0.6)}", color="cyan")

    def detect_regimes(self,
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None,
                      validate_economic_significance: bool = True,
                      validate_financial_relevance: bool = True) -> HybridRegimeResult:
        """
        Detect market regimes using hybrid NAS-TAS approach.

        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            validate_economic_significance: Whether to validate economic significance
            validate_financial_relevance: Whether to validate financial relevance

        Returns:
            HybridRegimeResult: Complete regime detection results
        """
        start_time = time.time()
        tprint("🚀 Starting hybrid NAS-TAS regime detection", color="blue")
        self.logger.info("🚀 Starting hybrid NAS-TAS regime detection...")

        try:
            # Step 1: Preprocess market data
            tprint("📊 Step 1: Preprocessing market data", color="cyan")
            processed_data = self._preprocess_market_data(market_data, timestamps)
            tprint(f"✅ Market data preprocessing completed: {processed_data.shape}")

            # Step 2: Extract features using both TAS and NAS approaches
            tprint("🔧 Step 2: Extracting TAS and NAS features", color="cyan")
            tas_features, tas_results = self._extract_tas_features(processed_data)
            tprint(f"✅ TAS features extracted: {tas_features.shape}")
            nas_features, nas_results = self._extract_nas_features(processed_data)
            tprint(f"✅ NAS features extracted: {nas_features.shape}")

            # Step 3: Combine features based on strategy
            tprint("🔄 Step 3: Combining features using strategy", color="cyan")
            combined_features = self._combine_features(
                tas_features, nas_features, tas_results, nas_results
            )
            tprint(f"✅ Features combined: {combined_features.shape}")

            # Step 4: Perform economic clustering on combined features
            tprint("🔍 Step 4: Performing economic clustering", color="cyan")
            economic_clustering_result = self._perform_economic_clustering(
                combined_features, processed_data
            )
            tprint(f"✅ Economic clustering completed: {economic_clustering_result.success}")

            if economic_clustering_result.success:
                tprint("✅ Economic clustering successful", color="green")
                regime_predictions = economic_clustering_result.regime_predictions
                regime_probabilities = economic_clustering_result.regime_probabilities
                cluster_metrics = economic_clustering_result.economic_clustering_metrics
                transition_probabilities = economic_clustering_result.transition_probabilities
                tprint(f"✅ Regime predictions: {len(np.unique(regime_predictions))} unique regimes")
            else:
                # Fallback to standard clustering
                tprint("⚠️ Economic clustering failed, using standard clustering fallback", color="yellow")
                regime_predictions, cluster_metrics = self._perform_standard_clustering(combined_features)
                tprint(f"✅ Standard clustering completed: {len(np.unique(regime_predictions))} regimes")
                regime_probabilities = self._calculate_regime_probabilities(
                    combined_features, regime_predictions
                )
                tprint("✅ Regime probabilities calculated")
                transition_probabilities = self._calculate_transition_probabilities(
                    regime_predictions, regime_probabilities
                )
                tprint("✅ Transition probabilities calculated")

            # Step 6: Evaluate economic and financial significance
            tprint("📊 Step 5: Evaluating economic and financial significance", color="cyan")
            economic_scores = np.zeros(self.config.n_regimes)
            financial_scores = np.zeros(self.config.n_regimes)
            stability_scores = np.zeros(self.config.n_regimes)

            if validate_economic_significance:
                tprint("💰 Evaluating economic significance", color="yellow")
                economic_scores = self._evaluate_economic_significance(
                    processed_data, regime_predictions, regime_probabilities
                )
                tprint(f"✅ Economic significance evaluation completed: {np.mean(economic_scores):.3f} average score")

            if validate_financial_relevance:
                tprint("💎 Evaluating financial relevance", color="yellow")
                financial_scores = self._evaluate_financial_relevance(
                    processed_data, regime_predictions, regime_probabilities
                )
                tprint(f"✅ Financial relevance evaluation completed: {np.mean(financial_scores):.3f} average score")

            # Calculate regime stability and momentum/volume scores
            tprint("⚖️ Calculating regime stability scores", color="yellow")
            stability_scores = self._calculate_regime_stability(
                regime_predictions, regime_probabilities, transition_probabilities
            )
            tprint(f"✅ Regime stability scores calculated: {np.mean(stability_scores):.3f} average stability")

            # Calculate momentum and volume profiles
            tprint("📈 Calculating momentum and volume profiles", color="yellow")
            momentum_scores = np.zeros(self.config.n_regimes)
            volume_profiles = np.zeros(self.config.n_regimes)

            if economic_clustering_result.success:
                momentum_scores = economic_clustering_result.momentum_scores
                volume_profiles = economic_clustering_result.volume_profiles
                tprint(f"✅ Momentum and volume profiles from economic clustering: {len(momentum_scores)} scores")
            else:
                tprint("⚠️ Using default momentum and volume profiles")

            # Perform coherent regime modeling
            if self.config.economic_evaluation.get('enabled', True):
                try:
                    tprint("📈 Performing coherent regime modeling", color="yellow")
                    regime_modeling_result = self.regime_modeler.model_regimes(
                        processed_data, regime_predictions, regime_probabilities
                    )
                    # Update economic scores with enhanced modeling
                    economic_scores = regime_modeling_result.economic_analysis.get('regime_significance_scores', economic_scores)
                    tprint("✅ Coherent regime modeling completed", color="green")
                except Exception as e:
                    self.logger.warning(f"Coherent regime modeling failed: {e}")
                    tprint(f"⚠️ Coherent regime modeling failed: {e}", color="yellow")

            # Step 7: Compile results
            tprint("📋 Step 6: Compiling final results", color="cyan")
            execution_time = time.time() - start_time

            # Economic clustering metrics
            economic_clustering_metrics = cluster_metrics if economic_clustering_result.success else {}

            result = HybridRegimeResult(
                success=True,
                regime_predictions=regime_predictions,
                regime_probabilities=regime_probabilities,
                economic_significance_scores=economic_scores,
                financial_relevance_scores=financial_scores,
                regime_stability_scores=stability_scores,
                transition_probabilities=transition_probabilities,
                combined_features=combined_features,
                tas_contributions=tas_results,
                nas_contributions=nas_results,
                clustering_metrics=cluster_metrics,
                economic_clustering_metrics=economic_clustering_metrics,
                momentum_scores=momentum_scores,
                volume_profiles=volume_profiles,
                execution_time=execution_time,
                metadata={
                    'combination_strategy': self.config.combination_strategy.value,
                    'n_regimes': self.config.n_regimes,
                    'data_points': len(processed_data),
                    'feature_dimensions': combined_features.shape[1] if combined_features.ndim > 1 else 1,
                    'timestamp': datetime.now().isoformat(),
                    'validation_performed': {
                        'economic': validate_economic_significance,
                        'financial': validate_financial_relevance
                    },
                    'economic_clustering_used': economic_clustering_result.success,
                    'momentum_integration': self.config.clustering_config.get('momentum_integration', True),
                    'volume_integration': self.config.clustering_config.get('volume_integration', True),
                    'position_aware_analysis': True
                }
            )

            self.logger.info("✅ Hybrid regime detection completed successfully")
            self.logger.info(f"   Execution time: {execution_time:.2f}s")
            self.logger.info(f"   Average economic significance: {np.mean(economic_scores):.3f}")
            self.logger.info(f"   Average financial relevance: {np.mean(financial_scores):.3f}")
            
            tprint("✅ Hybrid regime detection completed successfully", color="green")
            tprint(f"⏱️ Execution time: {execution_time:.2f}s", color="cyan")
            tprint(f"📊 Avg economic significance: {np.mean(economic_scores):.3f}, Avg financial relevance: {np.mean(financial_scores):.3f}", color="cyan")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Hybrid regime detection failed: {e}")
            tprint(f"❌ Hybrid regime detection failed: {e}", color="red")

            return HybridRegimeResult(
                success=False,
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                economic_significance_scores=np.array([]),
                financial_relevance_scores=np.array([]),
                regime_stability_scores=np.array([]),
                transition_probabilities=np.array([]),
                combined_features=np.array([]),
                tas_contributions={},
                nas_contributions={},
                clustering_metrics={},
                economic_clustering_metrics={},
                momentum_scores=np.array([]),
                volume_profiles=np.array([]),
                execution_time=execution_time,
                error_message=str(e)
            )

    def _preprocess_market_data(self,
                               market_data: Union[pd.DataFrame, np.ndarray],
                               timestamps: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Preprocess market data for regime detection."""
        try:
            tprint(f"🔧 [HYBRID_NAS_TAS] Preprocessing market data: {market_data.shape if hasattr(market_data, 'shape') else len(market_data)} points", color="cyan")
            tprint_debug(f"📊 [HYBRID_NAS_TAS] Input data type: {type(market_data)}")
            
            if isinstance(market_data, np.ndarray):
                tprint("🔄 [HYBRID_NAS_TAS] Converting numpy array to DataFrame", color="blue")
                # Convert numpy array to DataFrame with default columns
                columns = ['open', 'high', 'low', 'close', 'volume']
                if market_data.shape[1] >= 5:
                    market_data = pd.DataFrame(market_data[:, :5], columns=columns[:market_data.shape[1]])
                    tprint(f"✅ [HYBRID_NAS_TAS] Array converted with {market_data.shape[1]} columns", color="green")
                else:
                    market_data = pd.DataFrame(market_data, columns=columns[:market_data.shape[1]])
                    tprint(f"⚠️ [HYBRID_NAS_TAS] Array converted with {market_data.shape[1]} columns (limited)", color="yellow")

            if not isinstance(market_data, pd.DataFrame):
                tprint_error("❌ [HYBRID_NAS_TAS] Market data must be pandas DataFrame or numpy array")
                raise ValueError("Market data must be pandas DataFrame or numpy array")

            # Ensure required columns exist
            tprint("🔍 [HYBRID_NAS_TAS] Validating required columns", color="blue")
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = []
            for col in required_columns:
                if col not in market_data.columns:
                    if col == 'volume':
                        tprint(f"⚠️ [HYBRID_NAS_TAS] Volume column missing, using default value 1.0", color="yellow")
                        market_data[col] = 1.0  # Default volume
                    else:
                        missing_columns.append(col)
                        tprint_error(f"❌ [HYBRID_NAS_TAS] Required column '{col}' not found in market data")
            
            if missing_columns:
                raise ValueError(f"Required columns missing: {missing_columns}")
            
            tprint("✅ [HYBRID_NAS_TAS] All required columns validated", color="green")

            # Add timestamps if provided
            if timestamps is not None:
                tprint(f"📅 [HYBRID_NAS_TAS] Adding provided timestamps: {len(timestamps)} timestamps", color="blue")
                market_data['timestamp'] = timestamps
            elif 'timestamp' not in market_data.columns:
                tprint("📅 [HYBRID_NAS_TAS] No timestamps provided, generating default timestamps", color="yellow")
                market_data['timestamp'] = pd.date_range(
                    start=datetime.now().strftime('%Y-%m-%d'),
                    periods=len(market_data),
                    freq='1min'
                )
                tprint(f"✅ [HYBRID_NAS_TAS] Generated {len(market_data)} default timestamps", color="green")

            # Basic data cleaning
            tprint("🧹 [HYBRID_NAS_TAS] Cleaning data: removing NaN and infinite values", color="yellow")
            initial_rows = len(market_data)
            market_data = market_data.dropna()
            tprint(f"📊 [HYBRID_NAS_TAS] After NaN removal: {len(market_data)} rows (removed {initial_rows - len(market_data)})", color="cyan")
            
            market_data = market_data.replace([np.inf, -np.inf], np.nan).dropna()
            final_rows = len(market_data)
            tprint(f"📊 [HYBRID_NAS_TAS] After infinite value removal: {final_rows} rows (removed {initial_rows - final_rows})", color="cyan")
            tprint(f"✅ [HYBRID_NAS_TAS] Data cleaning completed: {final_rows} clean samples remaining", color="green")

            tprint(f"✅ [HYBRID_NAS_TAS] Data preprocessing complete: {final_rows} clean samples", color="green")
            return market_data

        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            tprint_error(f"❌ [HYBRID_NAS_TAS] Data preprocessing failed: {e}")
            raise

    def _extract_tas_features(self,
                             market_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract features using TAS (Tree Architecture Search) approach."""
        try:
            tprint("🌳 [HYBRID_NAS_TAS] Extracting TAS features...", color="blue")
            tprint_debug(f"📊 [HYBRID_NAS_TAS] Input market data shape: {market_data.shape}")
            self.logger.info("🔍 Extracting TAS features...")

            # Use TAS integration component
            tprint("🔧 [HYBRID_NAS_TAS] Using TAS integration component...")
            tprint_debug(f"🔧 [HYBRID_NAS_TAS] TAS config: {self.config.tas_config}")
            
            features, results = self.tas_integration.extract_features(market_data)
            tprint(f"✅ [HYBRID_NAS_TAS] TAS integration completed: {features.shape}")
            tprint_debug(f"📈 [HYBRID_NAS_TAS] TAS results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")

            self.logger.info(f"   TAS features extracted: {features.shape}")
            tprint_success(f"✅ [HYBRID_NAS_TAS] TAS features extracted: {features.shape}")
            tprint_performance(f"⚡ [HYBRID_NAS_TAS] TAS feature extraction performance: {features.shape[0]} samples, {features.shape[1]} features")
            return features, results

        except Exception as e:
            self.logger.warning(f"TAS feature extraction failed: {e}, using fallback")
            tprint_warning(f"⚠️ [HYBRID_NAS_TAS] TAS feature extraction failed: {e}, using fallback")
            tprint_debug(f"🔍 [HYBRID_NAS_TAS] TAS fallback reason: {str(e)}")
            # Fallback to basic feature extraction
            tprint("🔄 [HYBRID_NAS_TAS] Falling back to basic feature extraction", color="yellow")
            return self._extract_basic_features(market_data), {'method': 'fallback'}

    def _extract_nas_features(self,
                             market_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract features using NAS (Neural Architecture Search) approach."""
        try:
            tprint("🧠 [HYBRID_NAS_TAS] Extracting NAS features...", color="blue")
            tprint_debug(f"📊 [HYBRID_NAS_TAS] Input market data shape: {market_data.shape}")
            self.logger.info("🔍 Extracting NAS features...")

            # Use NAS integration component
            tprint("🔧 [HYBRID_NAS_TAS] Using NAS integration component...")
            tprint_debug(f"🔧 [HYBRID_NAS_TAS] NAS config: {self.config.nas_config}")
            
            features, results = self.nas_integration.extract_features(market_data)
            tprint(f"✅ [HYBRID_NAS_TAS] NAS integration completed: {features.shape}")
            tprint_debug(f"📈 [HYBRID_NAS_TAS] NAS results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")

            self.logger.info(f"   NAS features extracted: {features.shape}")
            tprint_success(f"✅ [HYBRID_NAS_TAS] NAS features extracted: {features.shape}")
            tprint_performance(f"⚡ [HYBRID_NAS_TAS] NAS feature extraction performance: {features.shape[0]} samples, {features.shape[1]} features")
            return features, results

        except Exception as e:
            self.logger.warning(f"NAS feature extraction failed: {e}, using fallback")
            tprint_warning(f"⚠️ [HYBRID_NAS_TAS] NAS feature extraction failed: {e}, using fallback")
            tprint_debug(f"🔍 [HYBRID_NAS_TAS] NAS fallback reason: {str(e)}")
            # Fallback to basic feature extraction
            tprint("🔄 [HYBRID_NAS_TAS] Falling back to basic feature extraction", color="yellow")
            return self._extract_basic_features(market_data), {'method': 'fallback'}

    def _combine_features(self,
                         tas_features: np.ndarray,
                         nas_features: np.ndarray,
                         tas_results: Dict[str, Any],
                         nas_results: Dict[str, Any]) -> np.ndarray:
        """Combine TAS and NAS features based on configured strategy."""
        try:
            tprint(f"🔄 [HYBRID_NAS_TAS] Combining TAS ({tas_features.shape}) and NAS ({nas_features.shape}) features", color="blue")
            tprint_debug(f"📊 [HYBRID_NAS_TAS] TAS features shape: {tas_features.shape}, NAS features shape: {nas_features.shape}")
            self.logger.info("🔄 Combining TAS and NAS features...")

            tas_weight = self.config.tas_config.get('weight', 0.5)
            nas_weight = self.config.nas_config.get('weight', 0.5)
            tprint_debug(f"⚖️ [HYBRID_NAS_TAS] Initial weights - TAS: {tas_weight}, NAS: {nas_weight}")

            # Normalize weights
            total_weight = tas_weight + nas_weight
            tas_weight = tas_weight / total_weight
            nas_weight = nas_weight / total_weight
            tprint(f"⚖️ [HYBRID_NAS_TAS] Normalized weights - TAS: {tas_weight:.3f}, NAS: {nas_weight:.3f}", color="cyan")

            # Strategy-specific combination
            strategy = self.config.combination_strategy
            tprint(f"🎯 [HYBRID_NAS_TAS] Using combination strategy: {strategy.value}", color="magenta")
            
            if strategy == RegimeCombinationStrategy.WEIGHTED_AVERAGE:
                tprint("📊 [HYBRID_NAS_TAS] Using weighted average combination", color="blue")
                combined_features = tas_weight * tas_features + nas_weight * nas_features
                tprint_debug(f"📈 [HYBRID_NAS_TAS] Weighted average: TAS*{tas_weight:.3f} + NAS*{nas_weight:.3f}")

            elif strategy == RegimeCombinationStrategy.ENSEMBLE_VOTING:
                tprint("🗳️ [HYBRID_NAS_TAS] Using ensemble voting combination", color="blue")
                # Use ensemble approach - take features with highest confidence
                tas_confidence = tas_results.get('confidence', 0.5)
                nas_confidence = nas_results.get('confidence', 0.5)
                tprint_debug(f"📊 [HYBRID_NAS_TAS] Confidence scores - TAS: {tas_confidence:.3f}, NAS: {nas_confidence:.3f}")

                if tas_confidence >= nas_confidence:
                    combined_features = tas_features
                    tprint("✅ [HYBRID_NAS_TAS] Selected TAS features (higher confidence)", color="green")
                else:
                    combined_features = nas_features
                    tprint("✅ [HYBRID_NAS_TAS] Selected NAS features (higher confidence)", color="green")

            elif strategy == RegimeCombinationStrategy.ECONOMIC_PRIORITY:
                tprint("💰 [HYBRID_NAS_TAS] Using economic priority combination", color="blue")
                # Prioritize features based on economic significance
                tas_economic = tas_results.get('economic_significance', 0.5)
                nas_economic = nas_results.get('economic_significance', 0.5)
                tprint_debug(f"💰 [HYBRID_NAS_TAS] Economic significance - TAS: {tas_economic:.3f}, NAS: {nas_economic:.3f}")

                if tas_economic >= nas_economic:
                    combined_features = tas_features * tas_weight + nas_features * nas_weight
                    tprint("✅ [HYBRID_NAS_TAS] TAS prioritized (higher economic significance)", color="green")
                else:
                    combined_features = nas_features * tas_weight + tas_features * nas_weight
                    tprint("✅ [HYBRID_NAS_TAS] NAS prioritized (higher economic significance)", color="green")

            elif strategy == RegimeCombinationStrategy.ADAPTIVE_FUSION:
                tprint("🧠 [HYBRID_NAS_TAS] Using adaptive fusion combination", color="blue")
                # Adaptive combination based on data characteristics
                combined_features = self._adaptive_feature_fusion(
                    tas_features, nas_features, tas_results, nas_results
                )
                tprint("✅ [HYBRID_NAS_TAS] Adaptive fusion completed", color="green")

            else:  # MULTI_OBJECTIVE
                tprint("🎯 [HYBRID_NAS_TAS] Using multi-objective combination", color="blue")
                # Concatenate features for multi-objective approach
                min_len = min(len(tas_features), len(nas_features))
                tas_subset = tas_features[:min_len]
                nas_subset = nas_features[:min_len]
                combined_features = np.hstack([tas_subset * tas_weight, nas_subset * nas_weight])
                tprint(f"📊 [HYBRID_NAS_TAS] Multi-objective: concatenated {min_len} samples with weights", color="cyan")

            self.logger.info(f"   Combined features shape: {combined_features.shape}")
            tprint_success(f"✅ [HYBRID_NAS_TAS] Feature combination complete: {combined_features.shape}")
            tprint_performance(f"⚡ [HYBRID_NAS_TAS] Combined features: {combined_features.shape[0]} samples, {combined_features.shape[1]} features")
            return combined_features

        except Exception as e:
            self.logger.error(f"Feature combination failed: {e}")
            tprint_error(f"❌ [HYBRID_NAS_TAS] Feature combination failed: {e}, using fallback")
            tprint_debug(f"🔍 [HYBRID_NAS_TAS] Combination fallback reason: {str(e)}")
            # Fallback to basic combination
            tprint("🔄 [HYBRID_NAS_TAS] Using basic combination fallback", color="yellow")
            return (tas_features + nas_features) / 2

    def _adaptive_feature_fusion(self,
                                tas_features: np.ndarray,
                                nas_features: np.ndarray,
                                tas_results: Dict[str, Any],
                                nas_results: Dict[str, Any]) -> np.ndarray:
        """Adaptively fuse features based on performance metrics."""
        try:
            # Calculate performance metrics for each feature set
            tas_performance = self._calculate_feature_performance(tas_features, tas_results)
            nas_performance = self._calculate_feature_performance(nas_features, nas_results)

            # Adapt weights based on performance
            total_performance = tas_performance + nas_performance
            if total_performance > 0:
                tas_weight = tas_performance / total_performance
                nas_weight = nas_performance / total_performance
            else:
                tas_weight = nas_weight = 0.5

            # Apply adaptive weights
            combined_features = tas_weight * tas_features + nas_weight * nas_features

            return combined_features

        except Exception as e:
            self.logger.warning(f"Adaptive fusion failed: {e}, using equal weights")
            return (tas_features + nas_features) / 2

    def _calculate_feature_performance(self, features: np.ndarray, results: Dict[str, Any]) -> float:
        """Calculate performance score for features."""
        try:
            performance = 1.0

            # Factor in confidence if available
            confidence = results.get('confidence', 0.5)
            performance *= confidence

            # Factor in economic significance if available
            economic = results.get('economic_significance', 0.5)
            performance *= economic

            return performance

        except:
            return 0.5

    def _perform_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform clustering on combined features."""
        try:
            tprint(f"🔍 [HYBRID_NAS_TAS] Performing clustering on {features.shape[0]} samples with {features.shape[1]} features", color="blue")
            tprint_debug(f"📊 [HYBRID_NAS_TAS] Features shape: {features.shape}, n_regimes: {self.config.n_regimes}")
            self.logger.info("🔍 Performing clustering on combined features...")

            # Use globally imported clustering algorithms
            algorithm = self.config.clustering_config.get('algorithm', 'adaptive')
            tprint(f"🎯 [HYBRID_NAS_TAS] Using clustering algorithm: {algorithm}", color="cyan")

            if algorithm == 'adaptive':
                tprint("🧠 [HYBRID_NAS_TAS] Using adaptive algorithm selection", color="blue")
                # Try different algorithms and choose best
                algorithms = {
                    'kmeans': KMeans(n_clusters=self.config.n_regimes, random_state=42),
                    'gmm': GaussianMixture(n_components=self.config.n_regimes, random_state=42),
                    'agglomerative': AgglomerativeClustering(n_clusters=self.config.n_regimes)
                }
                tprint_debug(f"🔧 [HYBRID_NAS_TAS] Testing {len(algorithms)} algorithms: {list(algorithms.keys())}")

                best_score = -1
                best_labels = None
                best_algorithm = None

                for name, alg in algorithms.items():
                    try:
                        tprint_debug(f"🧪 [HYBRID_NAS_TAS] Testing {name} algorithm...")
                        labels = alg.fit_predict(features)

                        # Calculate silhouette score
                        if len(set(labels)) > 1:
                            score = silhouette_score(features, labels)
                            tprint_debug(f"📊 [HYBRID_NAS_TAS] {name} silhouette score: {score:.3f}")
                        else:
                            score = 0.0
                            tprint_debug(f"⚠️ [HYBRID_NAS_TAS] {name} produced single cluster, score: 0.0")

                        if score > best_score:
                            best_score = score
                            best_labels = labels
                            best_algorithm = name
                            tprint(f"🏆 [HYBRID_NAS_TAS] New best algorithm: {name} (score: {score:.3f})", color="green")

                    except Exception as e:
                        tprint_debug(f"❌ [HYBRID_NAS_TAS] {name} algorithm failed: {e}")
                        continue

                if best_labels is None:
                    tprint_error("❌ [HYBRID_NAS_TAS] No clustering algorithm succeeded")
                    raise ValueError("No clustering algorithm succeeded")

                labels = best_labels
                tprint(f"✅ [HYBRID_NAS_TAS] Adaptive selection completed: {best_algorithm} (score: {best_score:.3f})", color="green")

            elif algorithm == 'kmeans':
                tprint("🔵 [HYBRID_NAS_TAS] Using K-Means clustering", color="blue")
                kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
                labels = kmeans.fit_predict(features)
                tprint("✅ [HYBRID_NAS_TAS] K-Means clustering completed", color="green")

            elif algorithm == 'gmm':
                tprint("🟢 [HYBRID_NAS_TAS] Using Gaussian Mixture Model clustering", color="blue")
                gmm = GaussianMixture(n_components=self.config.n_regimes, random_state=42)
                labels = gmm.fit_predict(features)
                tprint("✅ [HYBRID_NAS_TAS] GMM clustering completed", color="green")

            else:  # agglomerative
                tprint("🟡 [HYBRID_NAS_TAS] Using Agglomerative clustering", color="blue")
                agg = AgglomerativeClustering(n_clusters=self.config.n_regimes)
                labels = agg.fit_predict(features)
                tprint("✅ [HYBRID_NAS_TAS] Agglomerative clustering completed", color="green")

            # Calculate clustering metrics
            tprint("📊 [HYBRID_NAS_TAS] Calculating clustering metrics", color="blue")
            metrics = {}
            try:
                if len(set(labels)) > 1:
                    silhouette = silhouette_score(features, labels)
                    calinski = calinski_harabasz_score(features, labels)
                    metrics['silhouette_score'] = silhouette
                    metrics['calinski_harabasz_score'] = calinski
                    tprint(f"📈 [HYBRID_NAS_TAS] Silhouette score: {silhouette:.3f}", color="cyan")
                    tprint(f"📈 [HYBRID_NAS_TAS] Calinski-Harabasz score: {calinski:.3f}", color="cyan")
                else:
                    metrics['silhouette_score'] = 0.0
                    metrics['calinski_harabasz_score'] = 0.0
                    tprint("⚠️ [HYBRID_NAS_TAS] Single cluster detected, metrics set to 0.0", color="yellow")
            except Exception as e:
                tprint_warning(f"⚠️ [HYBRID_NAS_TAS] Metrics calculation failed: {e}")
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0

            unique_clusters = len(set(labels))
            self.logger.info(f"   Clustering completed with algorithm: {algorithm}")
            self.logger.info(f"   Silhouette score: {metrics.get('silhouette_score', 0):.3f}")
            
            tprint_success(f"✅ [HYBRID_NAS_TAS] Clustering completed: {algorithm}, {unique_clusters} clusters, silhouette: {metrics.get('silhouette_score', 0):.3f}")
            tprint_performance(f"⚡ [HYBRID_NAS_TAS] Clustering performance: {len(labels)} samples clustered into {unique_clusters} regimes")
            return labels, metrics

        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            tprint_error(f"❌ [HYBRID_NAS_TAS] Clustering failed: {e}, using random fallback")
            tprint_debug(f"🔍 [HYBRID_NAS_TAS] Clustering fallback reason: {str(e)}")
            # Fallback to simple clustering
            n_samples = len(features)
            labels = np.random.randint(0, self.config.n_regimes, n_samples)
            tprint_warning(f"⚠️ [HYBRID_NAS_TAS] Using random fallback: {n_samples} samples, {self.config.n_regimes} regimes")
            return labels, {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0}

    def _calculate_regime_probabilities(self,
                                      features: np.ndarray,
                                      labels: np.ndarray) -> np.ndarray:
        """Calculate probability of each data point belonging to each regime."""
        try:
            tprint("📊 Calculating regime probabilities using GMM", color="blue")
            
            # Use Gaussian Mixture Model to estimate probabilities
            gmm = GaussianMixture(n_components=self.config.n_regimes, random_state=42)
            gmm.fit(features)

            probabilities = gmm.predict_proba(features)
            
            # Ensure probabilities sum to 1 for each sample
            probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
            
            # Add small epsilon to avoid log(0) issues
            probabilities = np.clip(probabilities, 1e-10, 1.0)
            
            tprint(f"✅ Regime probabilities calculated: {probabilities.shape}", color="green")
            tprint(f"📈 Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]", color="cyan")
            
            return probabilities

        except Exception as e:
            self.logger.warning(f"Probability calculation failed: {e}")
            tprint_warning(f"⚠️ Probability calculation failed: {e}, using fallback")
            # Fallback to uniform probabilities
            n_samples = len(labels)
            uniform_prob = 1.0 / self.config.n_regimes
            probabilities = np.full((n_samples, self.config.n_regimes), uniform_prob)
            tprint("🔄 Using uniform probability fallback", color="yellow")
            return probabilities

    def _calculate_transition_probabilities(self,
                                          labels: np.ndarray,
                                          probabilities: np.ndarray) -> np.ndarray:
        """Calculate transition probabilities between regimes."""
        try:
            n_regimes = self.config.n_regimes

            # Calculate transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))

            for i in range(len(labels) - 1):
                current_regime = labels[i]
                next_regime = labels[i + 1]
                transition_matrix[current_regime, next_regime] += 1

            # Normalize to probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
            transition_matrix = transition_matrix / row_sums

            return transition_matrix

        except Exception as e:
            self.logger.warning(f"Transition probability calculation failed: {e}")
            # Fallback to uniform transition matrix
            return np.full((self.config.n_regimes, self.config.n_regimes),
                          1.0 / self.config.n_regimes)

    def _evaluate_economic_significance(self,
                                      market_data: pd.DataFrame,
                                      regime_labels: np.ndarray,
                                      regime_probabilities: np.ndarray) -> np.ndarray:
        """Evaluate economic significance of each regime."""
        try:
            self.logger.info("📊 Evaluating economic significance...")

            # Use economic evaluator component
            significance_scores = self.economic_evaluator.evaluate_regimes(
                market_data, regime_labels, regime_probabilities
            )

            self.logger.info(f"   Economic significance scores: {significance_scores}")
            return significance_scores

        except Exception as e:
            self.logger.warning(f"Economic evaluation failed: {e}")
            # Fallback to uniform scores
            return np.full(self.config.n_regimes, 0.5)

    def _evaluate_financial_relevance(self,
                                     market_data: pd.DataFrame,
                                     regime_labels: np.ndarray,
                                     regime_probabilities: np.ndarray) -> np.ndarray:
        """Evaluate financial relevance of each regime."""
        try:
            self.logger.info("💰 Evaluating financial relevance...")

            # Calculate financial metrics for each regime
            relevance_scores = []

            for regime_id in range(self.config.n_regimes):
                regime_mask = regime_labels == regime_id
                if np.sum(regime_mask) > 0:
                    regime_data = market_data[regime_mask]

                    # Calculate financial metrics
                    returns = regime_data['close'].pct_change().dropna()
                    if len(returns) > 0:
                        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
                        max_drawdown = self._calculate_max_drawdown(regime_data['close'])
                        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0

                        # Calculate composite financial relevance score
                        financial_score = (
                            0.4 * min(sharpe_ratio, 5.0) / 5.0 +  # Normalize sharpe ratio
                            0.3 * (1.0 - max_drawdown) +           # Lower drawdown is better
                            0.3 * win_rate                         # Higher win rate is better
                        )
                    else:
                        financial_score = 0.5
                else:
                    financial_score = 0.5

                relevance_scores.append(financial_score)

            self.logger.info(f"   Financial relevance scores: {relevance_scores}")
            return np.array(relevance_scores)

        except Exception as e:
            self.logger.warning(f"Financial evaluation failed: {e}")
            # Fallback to uniform scores
            return np.full(self.config.n_regimes, 0.5)

    def _calculate_max_drawdown(self, price_series: pd.Series) -> float:
        """Calculate maximum drawdown from price series."""
        try:
            peak = price_series.expanding().max()
            drawdown = (price_series - peak) / peak
            return abs(drawdown.min())

        except:
            return 0.0

    def _calculate_regime_stability(self,
                                   regime_labels: np.ndarray,
                                   regime_probabilities: np.ndarray,
                                   transition_matrix: np.ndarray) -> np.ndarray:
        """Calculate stability scores for each regime."""
        try:
            stability_scores = []

            for regime_id in range(self.config.n_regimes):
                # Calculate average probability for this regime
                regime_probs = regime_probabilities[:, regime_id]
                avg_prob = np.mean(regime_probs)

                # Calculate diagonal transition probability (staying in same regime)
                transition_stability = transition_matrix[regime_id, regime_id]

                # Calculate regime size stability
                regime_size = np.sum(regime_labels == regime_id)
                size_stability = min(regime_size / len(regime_labels), 1.0)

                # Combine stability metrics
                stability = (
                    0.5 * avg_prob +
                    0.3 * transition_stability +
                    0.2 * size_stability
                )

                stability_scores.append(stability)

            return np.array(stability_scores)

        except Exception as e:
            self.logger.warning(f"Stability calculation failed: {e}")
            return np.full(self.config.n_regimes, 0.5)

    def _extract_basic_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract basic features as fallback."""
        try:
            # Basic price features
            close_prices = market_data['close'].values.reshape(-1, 1)

            # Returns features
            returns = np.diff(close_prices.ravel(), prepend=close_prices[0])
            returns = returns.reshape(-1, 1)

            # Volatility features
            volatility = pd.Series(close_prices.ravel()).rolling(window=10, min_periods=1).std().values
            volatility = volatility.reshape(-1, 1)

            # Volume features
            volume = market_data.get('volume', np.ones(len(market_data))).values.reshape(-1, 1)

            # Combine features
            features = np.hstack([close_prices, returns, volatility, volume])

            # Remove NaN values
            mask = ~np.isnan(features).any(axis=1)
            features = features[mask]

            return features

        except Exception as e:
            self.logger.error(f"Basic feature extraction failed: {e}")
            # Return minimal features
            return market_data['close'].values.reshape(-1, 1)

    def _perform_economic_clustering(self, features: np.ndarray, market_data: pd.DataFrame) -> HybridRegimeResult:
        """Perform economic-aware clustering."""
        try:
            self.logger.info("🔍 Performing economic clustering...")

            # Use economic clusterer
            economic_result = self.economic_clusterer.cluster_economic_features(features, market_data)

            # Create result object
            result = HybridRegimeResult(
                success=True,
                regime_predictions=economic_result.labels,
                regime_probabilities=economic_result.probabilities,
                economic_significance_scores=economic_result.economic_significance,
                financial_relevance_scores=np.zeros(self.config.n_regimes),
                regime_stability_scores=np.zeros(self.config.n_regimes),
                transition_probabilities=economic_result.frontier_metrics.get('transition_matrix', np.zeros((self.config.n_regimes, self.config.n_regimes))),
                combined_features=features,
                tas_contributions={},
                nas_contributions={},
                clustering_metrics={},
                economic_clustering_metrics=economic_result.economic_metrics,
                momentum_scores=economic_result.momentum_scores,
                volume_profiles=economic_result.volume_profiles,
                execution_time=economic_result.execution_time,
                metadata={
                    'clustering_method': 'economic',
                    'algorithm_used': economic_result.algorithm_used,
                    'economic_features_used': True
                }
            )

            self.logger.info(f"   Economic clustering completed using {economic_result.algorithm_used}")
            return result

        except Exception as e:
            self.logger.warning(f"Economic clustering failed: {e}, using fallback")
            return HybridRegimeResult(
                success=False,
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                economic_significance_scores=np.array([]),
                financial_relevance_scores=np.array([]),
                regime_stability_scores=np.array([]),
                transition_probabilities=np.array([]),
                combined_features=features,
                tas_contributions={},
                nas_contributions={},
                clustering_metrics={},
                economic_clustering_metrics={},
                momentum_scores=np.array([]),
                volume_profiles=np.array([]),
                execution_time=0.0,
                error_message=str(e)
            )

    def _perform_standard_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform standard clustering as fallback."""
        try:
            self.logger.info("🔍 Performing standard clustering fallback...")

            # Use K-means as fallback
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
            labels = kmeans.fit_predict(features)

            # Calculate basic metrics
            metrics = {}

            try:
                if len(set(labels)) > 1:
                    metrics['silhouette_score'] = silhouette_score(features, labels)
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
            except:
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0

            self.logger.info("   Standard clustering completed")
            return labels, metrics

        except Exception as e:
            self.logger.error(f"Standard clustering failed: {e}")
            # Return random labels as last resort
            n_samples = len(features)
            labels = np.random.randint(0, self.config.n_regimes, n_samples)
            return labels, {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0}

# Convenience functions
def create_hybrid_regime_detector(config: Optional[HybridRegimeConfig] = None) -> HybridNASTASRegimeDetector:
    """Create a hybrid NAS-TAS regime detector."""
    tprint("🏭 Creating HybridNASTASRegimeDetector", color="blue")
    if config is None:
        config = HybridRegimeConfig()
        tprint("📋 Using default configuration", color="cyan")
    detector = HybridNASTASRegimeDetector(config)
    tprint("✅ HybridNASTASRegimeDetector created successfully", color="green")
    return detector

def quick_hybrid_regime_detection(market_data: Union[pd.DataFrame, np.ndarray],
                                 n_regimes: int = 8) -> HybridRegimeResult:
    """Quick hybrid regime detection with default settings."""
    tprint(f"⚡ Quick hybrid regime detection with {n_regimes} regimes", color="blue")
    config = HybridRegimeConfig(n_regimes=n_regimes)
    detector = HybridNASTASRegimeDetector(config)
    result = detector.detect_regimes(market_data)
    tprint(f"✅ Quick detection complete: success={result.success}", color="green")
    return result

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
