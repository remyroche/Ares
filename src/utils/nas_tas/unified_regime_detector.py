"""Unified NAS & TAS regime detection utilities."""

from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .unified_regime_config import RegimeDetectionMethod, UnifiedRegimeConfig
from .unified_result import UnifiedRegimeResult

try:  # pragma: no cover - optional colourful logging
    from src.utils.tprint import (
        tprint,
        tprint_debug,
        tprint_error,
        tprint_info,
        tprint_success,
        tprint_warning,
    )

    _TPRINT_AVAILABLE = True
except Exception:  # pragma: no cover - fallback used in tests
    _TPRINT_AVAILABLE = False

    def _plain_print(prefix: str, *args: Any) -> None:
        print(prefix, *args)

    def tprint(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        print(*args)

    def tprint_info(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        _plain_print("INFO:", *args)

    def tprint_debug(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        _plain_print("DEBUG:", *args)

    def tprint_warning(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        _plain_print("WARNING:", *args)

    def tprint_error(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        _plain_print("ERROR:", *args)

    def tprint_success(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        _plain_print("SUCCESS:", *args)


@dataclass
class DetectionPerformance:
    """Snapshot of a detection run used for lightweight telemetry."""

    timestamp: float
    detection_time: float
    n_samples: int
    n_regimes: int
    method: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedRegimeDetector:
    """High level orchestrator that blends NAS & TAS regime logic.

    The historical codebase contained a very large implementation in this module
    that ended up getting truncated which made the launcher crash at import
    time.  This rewritten version focuses on a reliable, lightweight core that
    produces deterministic yet meaningful placeholder regimes.  It keeps the
    public API intact so higher level training and monitoring components keep
    working while the team iterates on richer models.
    """

    def __init__(self, config: Optional[UnifiedRegimeConfig] = None) -> None:
        self.config = config or UnifiedRegimeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.performance_history: List[DetectionPerformance] = []
        self.last_result: Optional[UnifiedRegimeResult] = None

        tprint_info(
            "🚀 UnifiedRegimeDetector initialised",
            f"method={self.config.detection_method.value}",
            f"regimes={self.config.n_regimes}",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_regimes(
        self,
        market_data: Union[pd.DataFrame, np.ndarray, Iterable[Iterable[float]]],
        timestamps: Optional[Union[np.ndarray, Iterable[Any]]] = None,
    ) -> UnifiedRegimeResult:
        """Run the simplified unified regime detector.

        The implementation favours robustness over sophistication: it performs
        a couple of basic statistical transforms to produce regime labels and
        attaches quality metrics that downstream systems expect.  The method
        mirrors the signature used across the project so existing callers do
        not require any adjustments.
        """

        start_time = time.time()
        try:
            data_frame = self._normalise_market_data(market_data, timestamps)
            feature_frame = self._compute_features(data_frame)
            regime_labels, regime_probabilities = self._assign_regimes(feature_frame)

            economic_scores = self._compute_economic_significance(feature_frame)
            trading_scores = self._compute_trading_viability(feature_frame)
            stability_scores = self._compute_stability_scores(regime_labels)
            transition_probabilities = self._compute_transition_matrix(regime_labels)

            execution_time = time.time() - start_time

            result = UnifiedRegimeResult(
                success=True,
                regime_predictions=regime_labels,
                regime_probabilities=regime_probabilities,
                economic_significance_scores=economic_scores,
                trading_viability_scores=trading_scores,
                regime_stability_scores=stability_scores,
                transition_probabilities=transition_probabilities,
                micro_regimes=None,
                performance_metrics={
                    "detection_time": execution_time,
                    "n_samples": len(regime_labels),
                    "method": self.config.detection_method.value,
                },
                execution_time=execution_time,
                system_type="unified",
                architecture_used=self.config.detection_method.value,
                metadata=self._build_metadata(feature_frame, regime_labels, execution_time),
                error_message=None,
            )

            self._record_performance(result)
            self.last_result = result

            tprint_success(
                "✅ Unified regime detection completed",
                f"regimes_detected={len(np.unique(regime_labels))}",
                f"execution_time={execution_time:.3f}s",
            )
            return result
        except Exception as exc:  # pragma: no cover - defensive logging path
            execution_time = time.time() - start_time
            tprint_error(f"Unified regime detection failed: {exc}")
            self.logger.exception("Unified regime detection failure")
            failure = UnifiedRegimeResult(
                success=False,
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                economic_significance_scores=np.array([]),
                trading_viability_scores=np.array([]),
                regime_stability_scores=np.array([]),
                transition_probabilities=np.array([]),
                execution_time=execution_time,
                metadata={"error": str(exc)},
                error_message=str(exc),
            )
            self._record_performance(failure)
            self.last_result = failure
            return failure

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return aggregate statistics for recent detections."""

        if not self.performance_history:
            return {"total_runs": 0}

        detection_times = [p.detection_time for p in self.performance_history]
        success_rates = [1.0 if p.success else 0.0 for p in self.performance_history]

        metrics = {
            "total_runs": len(self.performance_history),
            "average_detection_time": float(np.mean(detection_times)),
            "fastest_detection_time": float(np.min(detection_times)),
            "slowest_detection_time": float(np.max(detection_times)),
            "success_rate": float(np.mean(success_rates)),
            "last_method": self.performance_history[-1].method,
            "last_samples": self.performance_history[-1].n_samples,
        }

        return metrics

    def save_results(self, result: UnifiedRegimeResult, filepath: Union[str, Path]) -> None:
        """Persist a detection result to JSON for reproducibility."""

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        serialisable = self._to_serialisable(result.to_dict())
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2)

        tprint_debug(f"💾 Unified regime result saved to {path}")

    def load_results(self, filepath: Union[str, Path]) -> UnifiedRegimeResult:
        """Load a previously persisted detection result."""

        path = Path(filepath)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        result = UnifiedRegimeResult.from_dict(data)
        self.last_result = result
        tprint_debug(f"📂 Unified regime result loaded from {path}")
        return result

    def summarize_result(self, result: Optional[UnifiedRegimeResult] = None) -> Dict[str, Any]:
        """Return a lightweight summary for dashboards and logs."""

        target = result or self.last_result
        if target is None:
            return {"success": False, "error": "no_result"}
        return target.get_summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_performance(self, result: UnifiedRegimeResult) -> None:
        detection_time = result.execution_time
        n_samples = int(len(result.regime_predictions))
        performance = DetectionPerformance(
            timestamp=time.time(),
            detection_time=float(detection_time),
            n_samples=n_samples,
            n_regimes=int(result.metadata.get("n_regimes", 0) if result.metadata else 0),
            method=self.config.detection_method.value,
            success=result.success,
            metadata=result.metadata or {},
        )
        self.performance_history.append(performance)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def _normalise_market_data(
        self,
        market_data: Union[pd.DataFrame, np.ndarray, Iterable[Iterable[float]]],
        timestamps: Optional[Union[np.ndarray, Iterable[Any]]],
    ) -> pd.DataFrame:
        if isinstance(market_data, pd.DataFrame):
            df = market_data.copy()
        else:
            array = np.asarray(list(market_data))
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            if array.shape[0] == 0:
                raise ValueError("Market data is empty")

            default_columns = ["open", "high", "low", "close", "volume"]
            column_count = min(array.shape[1], len(default_columns))
            columns = default_columns[:column_count]
            df = pd.DataFrame(array[:, :column_count], columns=columns)

        if timestamps is not None:
            ts_array = np.asarray(list(timestamps))
            if len(ts_array) != len(df):
                raise ValueError("Timestamp length does not match market data length")
            df = df.copy()
            df["timestamp"] = pd.to_datetime(ts_array)
        elif "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range(
                end=pd.Timestamp.utcnow(), periods=len(df), freq="min"
            )

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if numeric_columns.empty:
            raise ValueError("No numeric columns available for regime detection")

        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_df = df.select_dtypes(include=[np.number]).copy()
        if "close" not in feature_df.columns:
            first_numeric = feature_df.columns[0]
            feature_df["close"] = feature_df[first_numeric]

        feature_df["returns"] = feature_df["close"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        feature_df["volatility"] = (
            feature_df["returns"].rolling(window=10, min_periods=1).std().fillna(0.0)
        )
        feature_df["momentum"] = (
            feature_df["close"].pct_change(periods=5).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        )
        feature_df["trend_strength"] = (
            feature_df["momentum"].rolling(window=5, min_periods=1).mean().fillna(0.0)
        )
        feature_df["volume_zscore"] = (
            (feature_df.get("volume", feature_df["close"]).rolling(window=20, min_periods=1).mean())
        )
        feature_df = feature_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return feature_df

    def _assign_regimes(
        self, feature_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_regimes = max(2, int(self.config.n_regimes))
        returns = feature_df["returns"].to_numpy()

        quantiles = np.linspace(0, 100, num=n_regimes + 1)
        thresholds = np.percentile(returns, quantiles)
        # Avoid identical thresholds by adding a tiny epsilon when needed
        for idx in range(1, len(thresholds)):
            if thresholds[idx] <= thresholds[idx - 1]:
                thresholds[idx] = thresholds[idx - 1] + 1e-8

        regime_labels = np.digitize(returns, thresholds[1:-1], right=False).astype(int)
        centres = 0.5 * (thresholds[:-1] + thresholds[1:])
        probabilities = self._compute_probabilities(returns, centres)
        return regime_labels, probabilities

    def _compute_probabilities(self, returns: np.ndarray, centres: np.ndarray) -> np.ndarray:
        probabilities: List[np.ndarray] = []
        epsilon = 1e-8
        for value in returns:
            distances = np.abs(value - centres) + epsilon
            weights = 1.0 / distances
            probs = weights / np.sum(weights)
            probabilities.append(probs)
        return np.asarray(probabilities)

    def _compute_economic_significance(self, feature_df: pd.DataFrame) -> np.ndarray:
        returns = feature_df["returns"].to_numpy()
        mean = np.mean(returns)
        std = np.std(returns) + 1e-8
        z_scores = (returns - mean) / std
        scaled = 0.5 + 0.5 * np.tanh(z_scores)
        return np.clip(scaled, 0.0, 1.0)

    def _compute_trading_viability(self, feature_df: pd.DataFrame) -> np.ndarray:
        volatility = feature_df["volatility"].to_numpy()
        max_vol = np.max(volatility) + 1e-8
        viability = 1.0 - (volatility / max_vol)
        momentum = feature_df["momentum"].to_numpy()
        momentum_scaled = 0.5 + 0.5 * np.tanh(momentum)
        return np.clip(0.6 * viability + 0.4 * momentum_scaled, 0.0, 1.0)

    def _compute_stability_scores(self, regimes: np.ndarray) -> np.ndarray:
        stability = np.ones_like(regimes, dtype=float)
        streak = 1
        for idx in range(1, len(regimes)):
            if regimes[idx] == regimes[idx - 1]:
                streak += 1
            else:
                streak = 1
            stability[idx] = min(1.0, streak / max(self.config.min_regime_samples, 1))
        return stability

    def _compute_transition_matrix(self, regimes: np.ndarray) -> np.ndarray:
        n_regimes = max(2, int(self.config.n_regimes))
        matrix = np.zeros((n_regimes, n_regimes), dtype=float)
        for prev, curr in zip(regimes[:-1], regimes[1:]):
            if 0 <= prev < n_regimes and 0 <= curr < n_regimes:
                matrix[int(prev), int(curr)] += 1.0

        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
        return matrix

    def _build_metadata(
        self, feature_df: pd.DataFrame, regimes: np.ndarray, execution_time: float
    ) -> Dict[str, Any]:
        return {
            "n_regimes": int(len(np.unique(regimes))),
            "execution_time": float(execution_time),
            "detection_method": self.config.detection_method.value,
            "mean_return": float(feature_df["returns"].mean()),
            "std_return": float(feature_df["returns"].std()),
            "mean_volatility": float(feature_df["volatility"].mean()),
            "timestamp": time.time(),
        }

    def _to_serialisable(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {k: self._to_serialisable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_serialisable(v) for v in value]
        return value


__all__ = ["UnifiedRegimeDetector", "UnifiedRegimeResult"]
=======
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

class RegimeDetectionMethod(Enum):
    """Available regime detection methods."""
    CLUSTERING = "clustering"
    HIDDEN_MARKOV_MODEL = "hmm"
    CHANGE_POINT = "change_point"
    NEURAL_NETWORK = "neural_network"
    TREE_BASED = "tree_based"
    HYBRID = "hybrid"

class ArchitectureType(Enum):
    """Types of architectures supported."""
    NEURAL = "neural"
    TREE = "tree"
    HYBRID = "hybrid"

@dataclass
class RegimeDetectionConfig:
    """Configuration for regime detection."""
    
    # Core detection parameters
    method: RegimeDetectionMethod = RegimeDetectionMethod.HYBRID
    architecture_type: ArchitectureType = ArchitectureType.HYBRID
    n_regimes: int = 3
    min_regime_duration: int = 10
    
    # Clustering parameters
    clustering_algorithm: str = "kmeans"
    n_clusters: int = 3
    random_state: int = 42
    
    # HMM parameters
    hmm_n_states: int = 3
    hmm_n_iterations: int = 100
    
    # Change point detection parameters
    change_point_method: str = "pelt"
    change_point_penalty: float = 1.0
    
    # Neural network parameters
    neural_n_epochs: int = 100
    neural_hidden_size: int = 64
    
    # Tree parameters
    tree_max_depth: int = 10
    tree_min_samples_split: int = 2
    
    # Advanced parameters
    enable_regime_validation: bool = True
    stability_threshold: float = 0.7
    separation_threshold: float = 0.5

@dataclass
class RegimeInfo:
    """Information about a detected regime."""
    regime_id: int
    start_index: int
    end_index: int
    duration: int
    stability: float
    separation: float
    characteristics: Dict[str, Any]

@dataclass
class RegimeDetectionResult:
    """Result of regime detection."""
    
    # Core results
    regime_labels: np.ndarray
    regime_infos: List[RegimeInfo]
    n_regimes: int
    
    # Detection metadata
    method: RegimeDetectionMethod
    architecture_type: ArchitectureType
    detection_time: float
    
    # Quality metrics
    regime_quality_score: float
    stability_scores: List[float]
    separation_scores: List[float]
    
    # Metadata
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    success: bool = True
    error_message: Optional[str] = None

class UnifiedRegimeDetector:
    """
    Unified regime detector that consolidates all regime detection capabilities
    for both NAS and TAS systems.
    """
    
    def __init__(self, config: Optional[RegimeDetectionConfig] = None):
        """Initialize unified regime detector."""
        self.config = config or RegimeDetectionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance monitoring
        self.detection_history = []
        
        print(f"🚀 Unified Regime Detector initialized")
        print(f"   Method: {self.config.method.value}")
        print(f"   Architecture type: {self.config.architecture_type.value}")
        print(f"   Target regimes: {self.config.n_regimes}")
    
    def detect_regimes(self, 
                      data: pd.DataFrame,
                      features: Optional[List[str]] = None,
                      method: Optional[RegimeDetectionMethod] = None) -> RegimeDetectionResult:
        """
        Detect regimes in the data using the specified method.
        
        Args:
            data: Input data for regime detection
            features: List of feature columns to use (if None, uses all numeric columns)
            method: Detection method to use (defaults to config method)
            
        Returns:
            RegimeDetectionResult containing regime information
        """
        try:
            print("🔍 Starting regime detection...")
            start_time = time.time()
            
            # Use specified method or default from config
            detection_method = method or self.config.method
            
            # Validate inputs
            self._validate_inputs(data, features)
            
            # Prepare features
            feature_data = self._prepare_features(data, features)
            
            # Detect regimes using specified method
            regime_labels = self._detect_regimes_method(feature_data, detection_method)
            
            # Analyze detected regimes
            regime_infos = self._analyze_regimes(regime_labels, data)
            
            # Calculate quality metrics
            regime_quality_score = self._calculate_regime_quality(regime_infos)
            stability_scores = [ri.stability for ri in regime_infos]
            separation_scores = [ri.separation for ri in regime_infos]
            
            # Create result
            result = RegimeDetectionResult(
                regime_labels=regime_labels,
                regime_infos=regime_infos,
                n_regimes=len(regime_infos),
                method=detection_method,
                architecture_type=self.config.architecture_type,
                detection_time=time.time() - start_time,
                regime_quality_score=regime_quality_score,
                stability_scores=stability_scores,
                separation_scores=separation_scores
            )
            
            # Save detection history
            self.detection_history.append(result)
            
            print(f"✅ Regime detection completed in {result.detection_time:.2f}s")
            print(f"   Detected regimes: {result.n_regimes}")
            print(f"   Quality score: {regime_quality_score:.4f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Regime detection failed: {e}")
            return RegimeDetectionResult(
                regime_labels=np.array([]),
                regime_infos=[],
                n_regimes=0,
                method=detection_method or self.config.method,
                architecture_type=self.config.architecture_type,
                detection_time=0.0,
                regime_quality_score=0.0,
                stability_scores=[],
                separation_scores=[],
                success=False,
                error_message=str(e)
            )
    
    def _validate_inputs(self, data: pd.DataFrame, features: Optional[List[str]]):
        """Validate input data."""
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        
        if features is not None:
            for feature in features:
                if feature not in data.columns:
                    raise ValueError(f"Feature '{feature}' not found in data")
    
    def _prepare_features(self, data: pd.DataFrame, features: Optional[List[str]]) -> np.ndarray:
        """Prepare feature data for regime detection."""
        if features is None:
            # Use all numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_columns = features
        
        if len(numeric_columns) == 0:
            raise ValueError("No numeric features found for regime detection")
        
        # Extract and normalize features
        feature_data = data[numeric_columns].values
        
        # Handle missing values
        feature_data = np.nan_to_num(feature_data, nan=0.0)
        
        # Normalize features
        feature_means = np.mean(feature_data, axis=0)
        feature_stds = np.std(feature_data, axis=0)
        feature_stds[feature_stds == 0] = 1.0  # Avoid division by zero
        
        feature_data = (feature_data - feature_means) / feature_stds
        
        return feature_data
    
    def _detect_regimes_method(self, feature_data: np.ndarray, method: RegimeDetectionMethod) -> np.ndarray:
        """Detect regimes using the specified method."""
        if method == RegimeDetectionMethod.CLUSTERING:
            return self._detect_regimes_clustering(feature_data)
        elif method == RegimeDetectionMethod.TREE_BASED:
            return self._detect_regimes_tree(feature_data)
        elif method == RegimeDetectionMethod.NEURAL_NETWORK:
            return self._detect_regimes_neural(feature_data)
        elif method == RegimeDetectionMethod.HYBRID:
            return self._detect_regimes_hybrid(feature_data)
        else:
            raise ValueError(f"Unsupported regime detection method: {method}")
    
    def _detect_regimes_clustering(self, feature_data: np.ndarray) -> np.ndarray:
        """Detect regimes using clustering."""
        try:
            from sklearn.cluster import KMeans
            
            # Perform K-means clustering
            kmeans = KMeans(
                n_clusters=self.config.n_clusters,
                random_state=self.config.random_state,
                n_init=10
            )
            regime_labels = kmeans.fit_predict(feature_data)
            
            return regime_labels
            
        except ImportError:
            print("⚠️ scikit-learn not available, using simple clustering")
            return self._simple_clustering(feature_data)
    
    def _detect_regimes_tree(self, feature_data: np.ndarray) -> np.ndarray:
        """Detect regimes using tree-based methods."""
        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.cluster import KMeans
            
            # First perform clustering to get initial labels
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=self.config.random_state)
            initial_labels = kmeans.fit_predict(feature_data)
            
            # Use decision tree to refine regime boundaries
            tree = DecisionTreeClassifier(
                max_depth=self.config.tree_max_depth,
                min_samples_split=self.config.tree_min_samples_split,
                random_state=self.config.random_state
            )
            
            # Create features for tree (use rolling statistics)
            tree_features = self._create_tree_features(feature_data)
            
            # Fit tree on initial labels
            tree.fit(tree_features, initial_labels)
            
            # Predict refined labels
            regime_labels = tree.predict(tree_features)
            
            return regime_labels
            
        except ImportError:
            print("⚠️ scikit-learn not available, using simple clustering")
            return self._simple_clustering(feature_data)
    
    def _detect_regimes_neural(self, feature_data: np.ndarray) -> np.ndarray:
        """Detect regimes using neural network methods."""
        try:
            from sklearn.neural_network import MLPClassifier
            from sklearn.cluster import KMeans
            
            # First perform clustering to get initial labels
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=self.config.random_state)
            initial_labels = kmeans.fit_predict(feature_data)
            
            # Use neural network to refine regime boundaries
            neural_net = MLPClassifier(
                hidden_layer_sizes=(self.config.neural_hidden_size,),
                max_iter=self.config.neural_n_epochs,
                random_state=self.config.random_state
            )
            
            # Create features for neural network
            neural_features = self._create_neural_features(feature_data)
            
            # Fit neural network on initial labels
            neural_net.fit(neural_features, initial_labels)
            
            # Predict refined labels
            regime_labels = neural_net.predict(neural_features)
            
            return regime_labels
            
        except ImportError:
            print("⚠️ scikit-learn not available, using simple clustering")
            return self._simple_clustering(feature_data)
    
    def _detect_regimes_hybrid(self, feature_data: np.ndarray) -> np.ndarray:
        """Detect regimes using hybrid method."""
        # Combine multiple methods
        clustering_labels = self._detect_regimes_clustering(feature_data)
        tree_labels = self._detect_regimes_tree(feature_data)
        
        # Combine labels using voting
        combined_labels = np.zeros_like(clustering_labels)
        for i in range(len(combined_labels)):
            # Simple voting mechanism
            if clustering_labels[i] == tree_labels[i]:
                combined_labels[i] = clustering_labels[i]
            else:
                # Use clustering result as default
                combined_labels[i] = clustering_labels[i]
        
        return combined_labels
    
    def _simple_clustering(self, feature_data: np.ndarray) -> np.ndarray:
        """Simple clustering fallback when scikit-learn is not available."""
        # Simple distance-based clustering
        n_samples, n_features = feature_data.shape
        n_clusters = self.config.n_regimes
        
        # Initialize cluster centers randomly
        np.random.seed(self.config.random_state)
        cluster_centers = feature_data[np.random.choice(n_samples, n_clusters, replace=False)]
        
        regime_labels = np.zeros(n_samples, dtype=int)
        
        # Simple K-means iteration
        for iteration in range(10):  # Limited iterations
            # Assign points to closest cluster
            for i in range(n_samples):
                distances = [np.linalg.norm(feature_data[i] - center) for center in cluster_centers]
                regime_labels[i] = np.argmin(distances)
            
            # Update cluster centers
            new_centers = []
            for k in range(n_clusters):
                cluster_points = feature_data[regime_labels == k]
                if len(cluster_points) > 0:
                    new_centers.append(np.mean(cluster_points, axis=0))
                else:
                    new_centers.append(cluster_centers[k])
            
            cluster_centers = np.array(new_centers)
        
        return regime_labels
    
    def _create_tree_features(self, feature_data: np.ndarray) -> np.ndarray:
        """Create features for tree-based regime detection."""
        # Add rolling statistics as features
        window_size = min(10, len(feature_data) // 4)
        
        enhanced_features = [feature_data]  # Original features
        
        if window_size > 1:
            # Rolling mean
            rolling_mean = np.array([
                np.mean(feature_data[max(0, i-window_size):i+1], axis=0) 
                for i in range(len(feature_data))
            ])
            enhanced_features.append(rolling_mean)
            
            # Rolling std
            rolling_std = np.array([
                np.std(feature_data[max(0, i-window_size):i+1], axis=0) 
                for i in range(len(feature_data))
            ])
            enhanced_features.append(rolling_std)
        
        return np.concatenate(enhanced_features, axis=1)
    
    def _create_neural_features(self, feature_data: np.ndarray) -> np.ndarray:
        """Create features for neural network regime detection."""
        # Similar to tree features but with more transformations
        enhanced_features = [feature_data]  # Original features
        
        # Add lagged features
        if len(feature_data) > 1:
            lagged_features = np.vstack([feature_data[0], feature_data[:-1]])
            enhanced_features.append(lagged_features)
        
        # Add difference features
        if len(feature_data) > 1:
            diff_features = np.vstack([np.zeros_like(feature_data[0]), np.diff(feature_data, axis=0)])
            enhanced_features.append(diff_features)
        
        return np.concatenate(enhanced_features, axis=1)
    
    def _analyze_regimes(self, regime_labels: np.ndarray, data: pd.DataFrame) -> List[RegimeInfo]:
        """Analyze detected regimes."""
        regime_infos = []
        unique_regimes = np.unique(regime_labels)
        
        for regime_id in unique_regimes:
            regime_mask = regime_labels == regime_id
            regime_indices = np.where(regime_mask)[0]
            
            if len(regime_indices) < self.config.min_regime_duration:
                continue
            
            # Find regime boundaries
            regime_changes = np.diff(np.concatenate([[False], regime_mask, [False]]).astype(int))
            start_indices = np.where(regime_changes == 1)[0]
            end_indices = np.where(regime_changes == -1)[0]
            
            # Analyze each regime segment
            for start_idx, end_idx in zip(start_indices, end_indices):
                duration = end_idx - start_idx
                
                if duration >= self.config.min_regime_duration:
                    # Calculate regime characteristics
                    regime_data = data.iloc[start_idx:end_idx]
                    characteristics = self._calculate_regime_characteristics(regime_data)
                    
                    # Calculate stability and separation
                    stability = self._calculate_regime_stability(regime_labels, regime_id, start_idx, end_idx)
                    separation = self._calculate_regime_separation(regime_labels, regime_id, start_idx, end_idx)
                    
                    regime_info = RegimeInfo(
                        regime_id=int(regime_id),
                        start_index=start_idx,
                        end_index=end_idx,
                        duration=duration,
                        stability=stability,
                        separation=separation,
                        characteristics=characteristics
                    )
                    
                    regime_infos.append(regime_info)
        
        return regime_infos
    
    def _calculate_regime_characteristics(self, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate characteristics of a regime."""
        characteristics = {}
        
        # Basic statistics
        numeric_columns = regime_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in regime_data.columns:
                characteristics[f'{col}_mean'] = float(regime_data[col].mean())
                characteristics[f'{col}_std'] = float(regime_data[col].std())
                characteristics[f'{col}_min'] = float(regime_data[col].min())
                characteristics[f'{col}_max'] = float(regime_data[col].max())
        
        # Duration
        characteristics['duration'] = len(regime_data)
        
        return characteristics
    
    def _calculate_regime_stability(self, regime_labels: np.ndarray, regime_id: int, 
                                   start_idx: int, end_idx: int) -> float:
        """Calculate stability of a regime."""
        regime_segment = regime_labels[start_idx:end_idx]
        
        # Stability is the ratio of correct regime labels
        correct_labels = np.sum(regime_segment == regime_id)
        stability = correct_labels / len(regime_segment) if len(regime_segment) > 0 else 0.0
        
        return stability
    
    def _calculate_regime_separation(self, regime_labels: np.ndarray, regime_id: int, 
                                    start_idx: int, end_idx: int) -> float:
        """Calculate separation of a regime from other regimes."""
        # Count transitions at regime boundaries
        total_transitions = 0
        regime_transitions = 0
        
        if start_idx > 0:
            total_transitions += 1
            if regime_labels[start_idx-1] != regime_id:
                regime_transitions += 1
        
        if end_idx < len(regime_labels):
            total_transitions += 1
            if regime_labels[end_idx] != regime_id:
                regime_transitions += 1
        
        separation = regime_transitions / total_transitions if total_transitions > 0 else 1.0
        
        return separation
    
    def _calculate_regime_quality(self, regime_infos: List[RegimeInfo]) -> float:
        """Calculate overall regime quality score."""
        if not regime_infos:
            return 0.0
        
        # Combine stability and separation scores
        stability_scores = [ri.stability for ri in regime_infos]
        separation_scores = [ri.separation for ri in regime_infos]
        
        avg_stability = np.mean(stability_scores)
        avg_separation = np.mean(separation_scores)
        
        # Quality is combination of stability and separation
        quality_score = 0.6 * avg_stability + 0.4 * avg_separation
        
        return quality_score
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of regime detection performance."""
        if not self.detection_history:
            return {'total_detections': 0}
        
        recent_detections = self.detection_history[-10:]  # Last 10 detections
        
        summary = {
            'total_detections': len(self.detection_history),
            'avg_detection_time': np.mean([d.detection_time for d in recent_detections]),
            'avg_regime_quality': np.mean([d.regime_quality_score for d in recent_detections]),
            'avg_n_regimes': np.mean([d.n_regimes for d in recent_detections]),
            'success_rate': np.mean([d.success for d in recent_detections])
        }
        
        return summary

def create_unified_regime_detector(config: Optional[RegimeDetectionConfig] = None) -> UnifiedRegimeDetector:
    """Create a unified regime detector with specified configuration."""
    return UnifiedRegimeDetector(config)

__all__ = [
    'UnifiedRegimeDetector',
    'RegimeDetectionConfig',
    'RegimeDetectionResult',
    'RegimeInfo',
    'RegimeDetectionMethod',
    'ArchitectureType',
    'create_unified_regime_detector'
]
