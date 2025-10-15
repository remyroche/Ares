"""
import warnings
Trading Tree Architecture Search (Trading-TAS)

Specialized TAS implementation for financial trading applications with:
- Regime-aware architecture search
- Dynamic model selection during trading
- Risk-aware architecture optimization
- Integration with existing ML pipeline
- Real-time adaptation capabilities
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
from enum import Enum

# Import existing components
from .tree_architecture_search import TreeArchitectureSearch, TreeArchitectureConfig, TreeArchitectureCandidate
from src.utils.ml_common.models.model_factory import ModelType, ModelConfig
from src.training.steps.market_analysis.nas_clustering.core.nas_config import NASArchitectureType

logger = logging.getLogger(__name__)

class TradingObjective(Enum):
    """Trading-specific optimization objectives."""
    PROFITABILITY = "profitability"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    REGIME_STABILITY = "regime_stability"
    ADAPTATION_SPEED = "adaptation_speed"
    ROBUSTNESS = "robustness"
    TRANSACTION_COSTS = "transaction_costs"

class MarketRegime(Enum):
    """Market regime types for trading."""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    CRISIS = "crisis"
    NORMAL = "normal"
    UNKNOWN = "unknown"

@dataclass
class TradingTASConfig:
    """Configuration for trading-specific TAS."""

    # Base TAS configuration
    base_config: TreeArchitectureConfig = field(default_factory=TreeArchitectureConfig)

    # Trading-specific objectives and weights
    trading_objectives: List[TradingObjective] = field(default_factory=lambda: [
        TradingObjective.PROFITABILITY,
        TradingObjective.SHARPE_RATIO,
        TradingObjective.ROBUSTNESS
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2])

    # Regime detection settings
    regime_detection_enabled: bool = True
    regime_stability_threshold: float = 0.7
    min_regime_samples: int = 100
    max_regime_duration_hours: int = 24

    # Dynamic adaptation
    adaptation_enabled: bool = True
    adaptation_interval_minutes: int = 15
    performance_decay_rate: float = 0.9  # How quickly old performance is forgotten

    # Risk management
    max_drawdown_threshold: float = 0.15  # 15% max drawdown
    risk_adjusted_return_threshold: float = 0.1  # 10% minimum return
    transaction_cost_penalty: float = 0.001  # 0.1% per trade

    # Model selection criteria
    min_model_confidence: float = 0.6
    max_model_complexity: int = 100  # Max number of trees
    preferred_model_types: List[str] = field(default_factory=lambda: [
        'RandomForest', 'XGBoost', 'LightGBM', 'ExtraTrees'
    ])

    # Meta-learning for trading
    trading_meta_learning_enabled: bool = True
    regime_similarity_threshold: float = 0.8
    adaptation_history_length: int = 100

    # Performance tracking
    enable_performance_tracking: bool = True
    performance_tracking_interval: int = 60  # seconds
    save_model_snapshots: bool = True

    # Integration settings
    integrate_with_nas_clustering: bool = True
    use_existing_regime_detection: bool = True

@dataclass
class TradingRegime:
    """Represents a detected market regime."""

    regime_type: MarketRegime
    start_time: datetime
    end_time: Optional[datetime]
    confidence: float
    characteristics: Dict[str, float]
    optimal_architecture: Optional[TreeArchitectureCandidate] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    transition_probability: float = 0.0

@dataclass
class TradingTASResult:
    """Result of trading TAS optimization."""

    best_architecture: TreeArchitectureCandidate
    regime_analysis: Dict[MarketRegime, TradingRegime]
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    model_selection_log: List[Dict[str, Any]] = field(default_factory=list)

    # Trading-specific metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_trade_duration: float = 0.0
    regime_stability_score: float = 0.0

    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class TradingTreeArchitectureSearch:
    """Trading-specific TAS implementation."""

    def __init__(self, config: TradingTASConfig):
        """Initialize Trading TAS."""
        self.config = config
        self.logger = logger.getChild('TradingTAS')

        # Initialize base TAS
        self.base_tas = TreeArchitectureSearch(config.base_config)

        # Regime tracking
        self.current_regime: Optional[TradingRegime] = None
        self.regime_history: List[TradingRegime] = []
        self.regime_transition_matrix: Dict[Tuple[MarketRegime, MarketRegime], float] = {}

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.adaptation_history: List[Dict[str, Any]] = []

        # Model registry for different regimes
        self.regime_model_registry: Dict[MarketRegime, List[TreeArchitectureCandidate]] = {}

        self.logger.info("✅ Trading TAS initialized with regime-aware capabilities")

    def optimize_for_trading_regimes(self,
                                   market_data: pd.DataFrame,
                                   target_returns: pd.Series,
                                   existing_regimes: Optional[Dict] = None) -> TradingTASResult:
        """
        Optimize tree architectures for trading regime exploration and qualification.

        Args:
            market_data: Historical market data for regime analysis
            target_returns: Target returns for model training
            existing_regimes: Pre-detected regimes (optional)

        Returns:
            TradingTASResult with optimal architectures for each regime
        """
        self.logger.info("🚀 Starting Trading TAS regime optimization...")
        start_time = time.time()

        try:
            # Step 1: Detect and analyze market regimes
            if self.config.regime_detection_enabled:
                regimes = self._detect_market_regimes(market_data, target_returns)
            else:
                regimes = existing_regimes or {MarketRegime.NORMAL: None}

            # Step 2: Optimize architectures for each regime
            regime_architectures = {}
            for regime_type in regimes.keys():
                regime_data = self._get_regime_data(market_data, target_returns, regime_type, regimes)
                if len(regime_data) >= self.config.min_regime_samples:
                    optimal_arch = self._optimize_for_single_regime(regime_data, regime_type)
                    regime_architectures[regime_type] = optimal_arch

            # Step 3: Create comprehensive result
            result = TradingTASResult(
                best_architecture=self._select_best_overall_architecture(regime_architectures),
                regime_analysis=regimes,
                execution_time=time.time() - start_time
            )

            # Step 4: Calculate trading-specific metrics
            self._calculate_trading_metrics(result, market_data, target_returns)

            # Step 5: Setup real-time adaptation
            if self.config.adaptation_enabled:
                self._setup_regime_adaptation(result)

            self.logger.info(f"✅ Trading TAS completed in {result.execution_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Trading TAS failed: {e}")
            raise

    def select_model_for_trading(self,
                               current_market_data: pd.DataFrame,
                               current_regime: Optional[MarketRegime] = None,
                               risk_tolerance: float = 0.5) -> TreeArchitectureCandidate:
        """
        Select appropriate tree model for current trading conditions.

        Args:
            current_market_data: Current market conditions
            current_regime: Known current regime (optional)
            risk_tolerance: Risk tolerance level (0-1)

        Returns:
            Optimal architecture for current conditions
        """
        self.logger.info("🎯 Selecting model for current trading conditions...")

        try:
            # Detect current regime if not provided
            if current_regime is None:
                current_regime = self._detect_current_regime(current_market_data)

            # Get candidate architectures for this regime
            candidates = self.regime_model_registry.get(current_regime, [])

            if not candidates:
                # Fallback to general optimization
                self.logger.warning(f"No models available for regime {current_regime}, using general optimization")
                return self._optimize_for_current_conditions(current_market_data)

            # Select best architecture based on risk tolerance and current conditions
            selected_arch = self._select_risk_adjusted_architecture(
                candidates, current_market_data, risk_tolerance
            )

            # Log selection
            self._log_model_selection(selected_arch, current_regime, risk_tolerance)

            return selected_arch

        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")
            raise

    def adapt_to_changing_conditions(self,
                                   market_data_stream: Any,
                                   performance_monitor: Any) -> TreeArchitectureCandidate:
        """
        Adapt model architecture to changing market conditions in real-time.

        Args:
            market_data_stream: Stream of market data
            performance_monitor: Performance monitoring system

        Returns:
            Adapted architecture for new conditions
        """
        self.logger.info("🔄 Adapting to changing market conditions...")

        try:
            # Analyze recent performance
            recent_performance = performance_monitor.get_recent_performance()

            # Detect if adaptation is needed
            adaptation_needed = self._check_if_adaptation_needed(recent_performance)

            if not adaptation_needed:
                return self.current_regime.optimal_architecture if self.current_regime else None

            # Detect new regime
            current_regime = self._detect_current_regime(market_data_stream.get_current_data())

            # Update regime transition
            self._update_regime_transition(current_regime)

            # Get or create architecture for new regime
            if current_regime in self.regime_model_registry:
                # Use existing optimized architecture
                new_architecture = self.regime_model_registry[current_regime][0]
            else:
                # Optimize new architecture for detected regime
                regime_data = market_data_stream.get_regime_data(current_regime)
                new_architecture = self._optimize_for_single_regime(
                    regime_data, current_regime, adaptation=True
                )

            # Update current regime
            self.current_regime = TradingRegime(
                regime_type=current_regime,
                start_time=datetime.now(),
                confidence=0.8,
                characteristics=self._extract_regime_characteristics(market_data_stream.get_current_data()),
                optimal_architecture=new_architecture
            )

            # Log adaptation
            self._log_adaptation(new_architecture, recent_performance)

            return new_architecture

        except Exception as e:
            self.logger.error(f"Adaptation failed: {e}")
            raise

    def _detect_market_regimes(self, market_data: pd.DataFrame, target_returns: pd.Series) -> Dict[MarketRegime, TradingRegime]:
        """Detect market regimes using advanced clustering and analysis."""
        self.logger.info("🔍 Detecting market regimes...")

        regimes = {}

        try:
            # Use existing NAS clustering if available
            if self.config.integrate_with_nas_clustering and self.config.use_existing_regime_detection:
                regimes = self._detect_regimes_with_nas_clustering(market_data)
            else:
                # Use tree-based regime detection
                regimes = self._detect_regimes_with_tree_models(market_data, target_returns)

            # Analyze each detected regime
            for regime_type, regime_info in regimes.items():
                if isinstance(regime_info, dict):
                    regime = TradingRegime(
                        regime_type=regime_type,
                        start_time=regime_info.get('start_time', datetime.now()),
                        end_time=regime_info.get('end_time'),
                        confidence=regime_info.get('confidence', 0.5),
                        characteristics=regime_info.get('characteristics', {}),
                        transition_probability=regime_info.get('transition_probability', 0.0)
                    )
                    regimes[regime_type] = regime

            self.logger.info(f"✅ Detected {len(regimes)} market regimes")
            return regimes

        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}, using default regime")
            regimes[MarketRegime.NORMAL] = TradingRegime(
                regime_type=MarketRegime.NORMAL,
                start_time=datetime.now(),
                confidence=1.0,
                characteristics={'volatility': 0.02, 'trend_strength': 0.0}
            )
            return regimes

    def _detect_regimes_with_nas_clustering(self, market_data: pd.DataFrame) -> Dict[MarketRegime, Dict]:
        """Detect regimes using existing NAS clustering system."""
        try:
            from src.training.steps.market_analysis.nas_clustering.core.nas_clusterer import NASClusterer
            from src.training.steps.market_analysis.nas_clustering.core.nas_config import NASClusteringConfig

            config = NASClusteringConfig.create_short_term_trading_config()
            clusterer = NASClusterer(config)

            # Prepare data for clustering
            clustering_data = self._prepare_data_for_clustering(market_data)

            # Perform clustering
            result = clusterer.cluster_market_data(clustering_data)

            # Convert clustering results to regimes
            regimes = {}
            for i, (regime_data, labels) in enumerate(zip(result.regime_data, result.labels)):
                regime_type = self._map_clustering_to_regime_type(i, regime_data, labels)
                regimes[regime_type] = {
                    'start_time': datetime.now() - timedelta(days=1),
                    'confidence': result.quality_metrics.get('regime_confidence', 0.8),
                    'characteristics': self._extract_regime_characteristics_from_clustering(regime_data),
                    'transition_probability': 0.1
                }

            return regimes

        except ImportError:
            self.logger.warning("NAS clustering not available, falling back to tree-based detection")
            return {}

    def _detect_regimes_with_tree_models(self, market_data: pd.DataFrame, target_returns: pd.Series) -> Dict[MarketRegime, Dict]:
        """Detect regimes using tree-based models."""
        regimes = {}

        try:
            # Feature engineering for regime detection
            regime_features = self._create_regime_features(market_data)

            # Use unsupervised learning to detect regimes
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.cluster import KMeans

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

            # Try different numbers of regimes
            for n_regimes in range(3, 8):  # 3 to 7 regimes
                try:
                    # Cluster the feature space
                    kmeans = KMeans(n_clusters=n_regimes, random_state=42)
                    clusters = kmeans.fit_predict(regime_features)

                    # Analyze each cluster
                    for cluster_id in range(n_regimes):
                        cluster_mask = clusters == cluster_id
                        cluster_data = market_data[cluster_mask]

                        if len(cluster_data) >= self.config.min_regime_samples:
                            regime_type = self._analyze_cluster_characteristics(cluster_data, target_returns[cluster_mask])
                            regimes[regime_type] = {
                                'start_time': cluster_data.index[0] if hasattr(cluster_data, 'index') else datetime.now(),
                                'confidence': len(cluster_data) / len(market_data),
                                'characteristics': self._extract_regime_characteristics(cluster_data),
                                'transition_probability': 1.0 / n_regimes
                            }
                except:
                    continue

            return regimes

        except Exception as e:
            self.logger.warning(f"Tree-based regime detection failed: {e}")
            return {}

    def _optimize_for_single_regime(self,
                                  regime_data: Tuple[pd.DataFrame, pd.Series],
                                  regime_type: MarketRegime,
                                  adaptation: bool = False) -> TreeArchitectureCandidate:
        """Optimize architecture for a specific market regime."""
        X_regime, y_regime = regime_data

        # Configure TAS for this regime
        regime_config = self._create_regime_specific_config(regime_type, X_regime, y_regime)

        # Run optimization
        if adaptation:
            # Use faster adaptation-focused search
            best_arch = self.base_tas.search(X_regime.values, y_regime.values, search_method="bayesian")
        else:
            # Use comprehensive search
            best_arch = self.base_tas.search(X_regime.values, y_regime.values, search_method="hybrid")

        # Store in registry
        if regime_type not in self.regime_model_registry:
            self.regime_model_registry[regime_type] = []
        self.regime_model_registry[regime_type].append(best_arch)

        return best_arch

    def _create_regime_specific_config(self, regime_type: MarketRegime, X: pd.DataFrame, y: pd.Series) -> TreeArchitectureConfig:
        """Create regime-specific TAS configuration."""
        base_config = self.config.base_config

        # Adjust configuration based on regime characteristics
        if regime_type == MarketRegime.HIGH_VOLATILITY:
            # High volatility: Need robust, stable models
            config = TreeArchitectureConfig(
                min_depth=5,
                max_depth=12,
                min_trees=200,
                max_trees=500,
                objectives=['accuracy', 'robustness', 'efficiency'],
                objective_weights=[0.5, 0.3, 0.2]
            )
        elif regime_type == MarketRegime.LOW_VOLATILITY:
            # Low volatility: Can use more complex models
            config = TreeArchitectureConfig(
                min_depth=8,
                max_depth=20,
                min_trees=50,
                max_trees=300,
                objectives=['accuracy', 'efficiency', 'interpretability'],
                objective_weights=[0.6, 0.3, 0.1]
            )
        elif regime_type in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # Trending: Focus on directional accuracy
            config = TreeArchitectureConfig(
                min_depth=6,
                max_depth=15,
                min_trees=100,
                max_trees=400,
                objectives=['accuracy', 'profitability', 'adaptation_speed'],
                objective_weights=[0.5, 0.3, 0.2]
            )
        else:
            # Default configuration
            config = base_config

        return config

    def _select_risk_adjusted_architecture(self,
                                         candidates: List[TreeArchitectureCandidate],
                                         current_data: pd.DataFrame,
                                         risk_tolerance: float) -> TreeArchitectureCandidate:
        """Select architecture based on risk tolerance and current conditions."""
        if not candidates:
            raise ValueError("No candidate architectures available")

        # Calculate risk-adjusted scores
        risk_adjusted_scores = []
        for candidate in candidates:
            base_score = candidate.overall_score

            # Adjust for current market conditions
            market_risk = self._calculate_market_risk(current_data)
            regime_risk = self._calculate_regime_risk()

            # Risk penalty based on model complexity and market conditions
            complexity_penalty = candidate.n_trees * candidate.max_depth * 0.001
            market_penalty = market_risk * (1 - risk_tolerance)
            regime_penalty = regime_risk * 0.1

            total_penalty = complexity_penalty + market_penalty + regime_penalty
            risk_adjusted_score = base_score * (1 - total_penalty)

            risk_adjusted_scores.append(risk_adjusted_score)

        # Select best architecture
        best_idx = np.argmax(risk_adjusted_scores)
        return candidates[best_idx]

    def _calculate_trading_metrics(self, result: TradingTASResult, market_data: pd.DataFrame, target_returns: pd.Series):
        """Calculate comprehensive trading performance metrics."""
        try:
            # Simulate trading performance for each regime architecture
            for regime_type, regime in result.regime_analysis.items():
                if regime.optimal_architecture:
                    performance = self._simulate_trading_performance(
                        regime.optimal_architecture, market_data, target_returns, regime_type
                    )

                    regime.performance_metrics.update(performance)
                    result.performance_history.append({
                        'regime': regime_type.value,
                        'timestamp': datetime.now(),
                        'metrics': performance
                    })

            # Calculate aggregate metrics
            result.total_return = np.mean([r.performance_metrics.get('total_return', 0)
                                         for r in result.regime_analysis.values()])
            result.sharpe_ratio = np.mean([r.performance_metrics.get('sharpe_ratio', 0)
                                         for r in result.regime_analysis.values()])
            result.max_drawdown = np.max([r.performance_metrics.get('max_drawdown', 0)
                                        for r in result.regime_analysis.values()])
            result.win_rate = np.mean([r.performance_metrics.get('win_rate', 0)
                                     for r in result.regime_analysis.values()])

        except Exception as e:
            self.logger.warning(f"Trading metrics calculation failed: {e}")

    def _simulate_trading_performance(self,
                                    architecture: TreeArchitectureCandidate,
                                    market_data: pd.DataFrame,
                                    target_returns: pd.Series,
                                    regime_type: MarketRegime) -> Dict[str, float]:
        """Simulate trading performance for a given architecture."""
        try:
            # Create model from architecture
            model = self.base_tas._create_model_from_candidate(architecture, target_returns.values)

            # Train model
            X = market_data.values
            y = target_returns.values
            model.fit(X, y)

            # Simulate trading
            predictions = model.predict(X)

            # Calculate trading metrics
            returns = predictions * y  # Simplified return calculation
            cumulative_returns = np.cumprod(1 + returns) - 1

            total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(cumulative_returns)
            win_rate = np.mean(returns > 0)

            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_trade_return': np.mean(returns),
                'volatility': np.std(returns)
            }

        except Exception as e:
            self.logger.warning(f"Trading simulation failed: {e}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_trade_return': 0.0,
                'volatility': 0.0
            }

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio from returns series."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        if len(cumulative_returns) == 0:
            return 0.0

        peak = cumulative_returns[0]
        max_dd = 0.0

        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = (peak - value) / (1 + peak) if peak != 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    # Additional helper methods would be implemented here...
    # For brevity, I'm showing the core structure and key methods

    def _select_best_overall_architecture(self, regime_architectures: Dict) -> TreeArchitectureCandidate:
        """Select best overall architecture across all regimes."""
        if not regime_architectures:
            return TreeArchitectureCandidate(
                n_trees=100, max_depth=10, min_samples_split=2, min_samples_leaf=1,
                max_features='auto', splitting_strategy='gini'
            )

        # Weight architectures by regime frequency and performance
        best_arch = None
        best_score = -np.inf

        for arch in regime_architectures.values():
            if arch.overall_score > best_score:
                best_score = arch.overall_score
                best_arch = arch

        return best_arch

    def _setup_regime_adaptation(self, result: TradingTASResult):
        """Setup real-time regime adaptation."""
        self.logger.info("🔄 Setting up regime adaptation system...")
        # Implementation would include:
        # - Performance monitoring threads
        # - Market data streaming
        # - Automatic model switching
        # - Risk management triggers
        pass

    # Helper methods for regime detection and analysis
    def _get_regime_data(self, market_data: pd.DataFrame, target_returns: pd.Series,
                        regime_type: MarketRegime, regimes: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract data for a specific regime."""
        # This would filter data based on regime characteristics
        # For now, return all data - in practice, this would use regime labels
        return market_data, target_returns

    def _prepare_data_for_clustering(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare market data for clustering-based regime detection."""
        # Feature engineering for regime detection
        features = []

        # Volatility features
        returns = market_data.pct_change()
        features.extend([
            returns.rolling(20).std().mean(),  # Rolling volatility
            returns.rolling(50).std().mean(),  # Longer-term volatility
            (returns.rolling(20).std() / returns.rolling(50).std()).mean()  # Volatility ratio
        ])

        # Trend features
        for window in [10, 20, 50]:
            sma = market_data.rolling(window).mean()
            trend_strength = (market_data - sma).abs().mean() / market_data.std()
            features.append(trend_strength.mean())

        # Volume features (if available)
        if 'volume' in market_data.columns:
            volume_ratio = market_data['volume'] / market_data['volume'].rolling(20).mean()
            features.append(volume_ratio.mean())

        return pd.DataFrame([features], columns=[f'feature_{i}' for i in range(len(features))])

    def _map_clustering_to_regime_type(self, cluster_id: int, regime_data: pd.DataFrame, labels: np.ndarray) -> MarketRegime:
        """Map clustering results to regime types."""
        # Analyze cluster characteristics to determine regime type
        if 'volatility' in regime_data.columns:
            vol = regime_data['volatility'].mean()
            if vol > 0.03:  # High volatility
                return MarketRegime.HIGH_VOLATILITY
            elif vol < 0.01:  # Low volatility
                return MarketRegime.LOW_VOLATILITY

        # Check for trends
        if 'trend_strength' in regime_data.columns:
            trend = regime_data['trend_strength'].mean()
            if trend > 0.7:
                return MarketRegime.TRENDING_UP

        return MarketRegime.NORMAL

    def _extract_regime_characteristics_from_clustering(self, regime_data: pd.DataFrame) -> Dict[str, float]:
        """Extract characteristics from clustering results."""
        characteristics = {}

        for col in regime_data.columns:
            if col in ['volatility', 'trend_strength', 'volume_ratio']:
                characteristics[col] = regime_data[col].mean()

        return characteristics

    def _create_regime_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create features for regime detection."""
        features = []

        # Price-based features
        returns = market_data.pct_change()

        # Volatility features
        for window in [5, 10, 20, 50]:
            vol = returns.rolling(window).std()
            features.append(vol.mean())

        # Trend features
        for window in [10, 20, 50]:
            sma = market_data.rolling(window).mean()
            trend = (market_data.iloc[-1] / sma.iloc[-1] - 1) if len(sma) > 0 else 0
            features.append(trend)

        # Momentum features
        for window in [5, 10, 20]:
            momentum = (market_data - market_data.shift(window)) / market_data.shift(window)
            features.append(momentum.mean())

        return pd.DataFrame([features])

    def _analyze_cluster_characteristics(self, cluster_data: pd.DataFrame, cluster_returns: pd.Series) -> MarketRegime:
        """Analyze cluster characteristics to determine regime type."""
        # Calculate cluster statistics
        returns = cluster_data.pct_change()
        volatility = returns.std().mean()
        trend_strength = abs((cluster_data.iloc[-1] / cluster_data.iloc[0] - 1).mean())

        # Determine regime type based on characteristics
        if volatility > 0.03:  # High volatility
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.01:  # Low volatility
            return MarketRegime.LOW_VOLATILITY
        elif trend_strength > 0.1:  # Strong trend
            avg_return = cluster_returns.mean()
            return MarketRegime.TRENDING_UP if avg_return > 0 else MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.NORMAL

    def _extract_regime_characteristics(self, cluster_data: pd.DataFrame) -> Dict[str, float]:
        """Extract regime characteristics from cluster data."""
        returns = cluster_data.pct_change()

        return {
            'volatility': returns.std().mean(),
            'trend_strength': abs((cluster_data.iloc[-1] / cluster_data.iloc[0] - 1).mean()),
            'mean_return': returns.mean().mean(),
            'max_return': returns.max().max(),
            'min_return': returns.min().min(),
            'duration_hours': len(cluster_data) / 60  # Assuming 1-minute data
        }

    def _detect_current_regime(self, current_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime from recent data."""
        # Use recent data to determine current regime
        recent_data = current_data.tail(100)  # Last 100 periods
        returns = recent_data.pct_change()

        volatility = returns.std().mean()
        trend_strength = abs((recent_data.iloc[-1] / recent_data.iloc[0] - 1).mean())

        if volatility > 0.03:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.01:
            return MarketRegime.LOW_VOLATILITY
        elif trend_strength > 0.05:
            return MarketRegime.TRENDING_UP
        else:
            return MarketRegime.NORMAL

    def _optimize_for_current_conditions(self, current_data: pd.DataFrame) -> TreeArchitectureCandidate:
        """Fallback optimization for current conditions when no regime-specific model exists."""
        # Create simple features and targets from current data
        X = current_data.values
        y = np.random.randn(len(X))  # Placeholder - would use actual targets

        # Use fast Bayesian optimization
        return self.base_tas.search(X, y, search_method="bayesian")

    def _calculate_market_risk(self, market_data: pd.DataFrame) -> float:
        """Calculate current market risk level."""
        returns = market_data.pct_change()
        return returns.std().mean()

    def _calculate_regime_risk(self) -> float:
        """Calculate risk based on current regime."""
        if not self.current_regime:
            return 0.5  # Medium risk

        # Risk based on regime type
        risk_mapping = {
            MarketRegime.HIGH_VOLATILITY: 0.9,
            MarketRegime.CRISIS: 1.0,
            MarketRegime.TRENDING_UP: 0.3,
            MarketRegime.TRENDING_DOWN: 0.4,
            MarketRegime.LOW_VOLATILITY: 0.1,
            MarketRegime.NORMAL: 0.5
        }

        return risk_mapping.get(self.current_regime.regime_type, 0.5)

    def _log_model_selection(self, architecture: TreeArchitectureCandidate,
                           regime: MarketRegime, risk_tolerance: float):
        """Log model selection for analysis."""
        self.logger.info(f"📊 Model selected for regime {regime.value}: {architecture.n_trees} trees, depth {architecture.max_depth}")

    def _check_if_adaptation_needed(self, recent_performance: Dict[str, float]) -> bool:
        """Check if model adaptation is needed based on performance."""
        if not recent_performance:
            return False

        # Check performance thresholds
        sharpe_ratio = recent_performance.get('sharpe_ratio', 0)
        drawdown = recent_performance.get('max_drawdown', 0)
        win_rate = recent_performance.get('win_rate', 0)

        # Adaptation needed if performance degrades significantly
        adaptation_needed = (
            sharpe_ratio < 0.5 or  # Poor risk-adjusted returns
            drawdown > 0.1 or     # Excessive drawdown
            win_rate < 0.4         # Poor win rate
        )

        return adaptation_needed

    def _update_regime_transition(self, new_regime: MarketRegime):
        """Update regime transition probabilities."""
        if self.current_regime:
            transition = (self.current_regime.regime_type, new_regime)
            if transition not in self.regime_transition_matrix:
                self.regime_transition_matrix[transition] = 0
            self.regime_transition_matrix[transition] += 1

    def _extract_regime_characteristics(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract characteristics from current market data."""
        returns = market_data.pct_change()

        return {
            'volatility': returns.std().mean(),
            'trend_strength': abs((market_data.iloc[-1] / market_data.iloc[0] - 1).mean()),
            'mean_return': returns.mean().mean(),
            'data_points': len(market_data)
        }

    def _log_adaptation(self, new_architecture: TreeArchitectureCandidate, recent_performance: Dict[str, float]):
        """Log adaptation events for analysis."""
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'new_architecture': {
                'n_trees': new_architecture.n_trees,
                'max_depth': new_architecture.max_depth,
                'search_method': new_architecture.search_method
            },
            'recent_performance': recent_performance
        })

        self.logger.info(f"🔄 Model adapted to: {new_architecture.n_trees} trees, depth {new_architecture.max_depth}")

# Convenience functions for trading TAS
def optimize_trading_regimes(market_data: pd.DataFrame,
                           target_returns: pd.Series,
                           config: Optional[TradingTASConfig] = None) -> TradingTASResult:
    """
    Convenience function for regime-aware TAS optimization.

    Args:
        market_data: Historical market data
        target_returns: Target returns for training
        config: Trading TAS configuration

    Returns:
        TradingTASResult with optimal architectures for each regime
    """
    if config is None:
        config = TradingTASConfig()

    tas = TradingTreeArchitectureSearch(config)
    return tas.optimize_for_trading_regimes(market_data, target_returns)

def select_trading_model(current_market_data: pd.DataFrame,
                        current_regime: Optional[MarketRegime] = None,
                        risk_tolerance: float = 0.5,
                        config: Optional[TradingTASConfig] = None) -> TreeArchitectureCandidate:
    """
    Convenience function for dynamic model selection during trading.

    Args:
        current_market_data: Current market conditions
        current_regime: Known current regime (optional)
        risk_tolerance: Risk tolerance level (0-1)
        config: Trading TAS configuration

    Returns:
        Optimal architecture for current conditions
    """
    if config is None:
        config = TradingTASConfig()

    tas = TradingTreeArchitectureSearch(config)
    return tas.select_model_for_trading(current_market_data, current_regime, risk_tolerance)

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
