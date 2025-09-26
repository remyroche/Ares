"""
Regime Detection for TAS

Comprehensive regime detection system for tree architecture search including
unsupervised clustering, regime qualification, and regime transition analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import existing regime detection components
try:
    from src.utils.nas_tas.unsupervised_regime_detection import UnsupervisedRegimeDetector, RegimeDetectionConfig
    from ..regime_analysis.regime_qualification import RegimeQualifier, RegimeQualificationConfig
    REGIME_ANALYSIS_AVAILABLE = True
except ImportError:
    REGIME_ANALYSIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RegimeDetectionMethod(Enum):
    """Regime detection methods."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    GMM = "gmm"
    HMM = "hmm"
    UNSUPERVISED = "unsupervised"
    SUPERVISED = "supervised"


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    
    # Detection method
    detection_method: RegimeDetectionMethod = RegimeDetectionMethod.UNSUPERVISED
    
    # Regime parameters
    n_regimes: int = 5
    min_regime_duration: int = 20
    max_regimes: int = 20
    
    # Clustering parameters
    clustering_algorithm: str = "kmeans"  # "kmeans", "dbscan", "gmm", "hmm"
    n_clusters_range: Tuple[int, int] = (2, 20)
    silhouette_threshold: float = 0.3
    
    # Regime qualification
    enable_regime_qualification: bool = True
    qualification_threshold: float = 0.6
    economic_significance: bool = True
    trading_viability: bool = True
    
    # Regime analysis
    enable_regime_analysis: bool = True
    regime_stability: bool = True
    regime_transitions: bool = True
    regime_persistence: bool = True
    
    # Feature engineering for regimes
    enable_regime_features: bool = True
    regime_feature_types: List[str] = field(default_factory=lambda: [
        'volatility', 'trend', 'volume', 'momentum', 'volatility_of_volatility'
    ])
    
    # Output configuration
    save_regime_data: bool = True
    output_directory: str = "regime_data"
    cache_regimes: bool = True


@dataclass
class RegimeResult:
    """Result of regime detection."""
    
    # Regime data
    regime_labels: np.ndarray
    regime_centers: Dict[int, np.ndarray]
    regime_statistics: Dict[int, Dict[str, Any]]
    qualified_regimes: Dict[str, Any]
    
    # Regime analysis
    regime_transitions: List[Dict[str, Any]]
    regime_stability: Dict[int, float]
    regime_persistence: Dict[int, float]
    regime_quality_scores: Dict[int, float]
    
    # Regime features
    regime_features: pd.DataFrame
    regime_feature_importance: Dict[str, float]
    regime_feature_correlations: pd.DataFrame
    
    # Detection metadata
    detection_method: str
    detection_time: float
    n_regimes_detected: int
    n_qualified_regimes: int
    regime_quality_score: float
    
    # Performance metrics
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    
    # Metadata
    config: RegimeConfig
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class RegimeDetector:
    """
    Comprehensive regime detector for TAS.
    
    Provides regime detection, qualification, and analysis
    for tree architecture search.
    """
    
    def __init__(self, config: RegimeConfig):
        """Initialize regime detector.
        
        Args:
            config: Regime detection configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize regime detection components
        self.regime_detector = None
        self.regime_qualifier = None
        
        # Initialize available components
        self._initialize_components()
        
        self.logger.info("✅ Regime Detector initialized")
        self.logger.info(f"📊 Detection method: {config.detection_method.value}")
        self.logger.info(f"📊 Number of regimes: {config.n_regimes}")
        self.logger.info(f"📊 Qualification enabled: {config.enable_regime_qualification}")
    
    def _initialize_components(self):
        """Initialize available regime detection components."""
        if REGIME_ANALYSIS_AVAILABLE:
            try:
                # Initialize regime detector
                regime_detection_config = RegimeDetectionConfig(
                    n_regimes=self.config.n_regimes,
                    min_regime_duration=self.config.min_regime_duration,
                    clustering_algorithm=self.config.clustering_algorithm,
                    n_clusters_range=self.config.n_clusters_range,
                    silhouette_threshold=self.config.silhouette_threshold
                )
                self.regime_detector = UnsupervisedRegimeDetector(regime_detection_config)
                
                # Initialize regime qualifier
                if self.config.enable_regime_qualification:
                    regime_qualification_config = RegimeQualificationConfig(
                        qualification_threshold=self.config.qualification_threshold,
                        economic_significance=self.config.economic_significance,
                        trading_viability=self.config.trading_viability
                    )
                    self.regime_qualifier = RegimeQualifier(regime_qualification_config)
                
                self.logger.info("✅ Regime analysis components initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Regime analysis components not available: {e}")
    
    def detect_regimes(self, data: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> RegimeResult:
        """
        Detect regimes in data.
        
        Args:
            data: Input data
            features: Optional engineered features
            
        Returns:
            Regime detection result
        """
        self.logger.info("🚀 Starting regime detection")
        start_time = datetime.now()
        
        try:
            # Prepare data for regime detection
            regime_data = self._prepare_regime_data(data, features)
            
            # Detect regimes
            if self.config.detection_method == RegimeDetectionMethod.UNSUPERVISED and self.regime_detector:
                regime_labels, regime_centers, regime_statistics = self._detect_unsupervised_regimes(regime_data)
            else:
                regime_labels, regime_centers, regime_statistics = self._detect_basic_regimes(regime_data)
            
            # Qualify regimes
            qualified_regimes = {}
            if self.config.enable_regime_qualification and self.regime_qualifier:
                qualification_result = self.regime_qualifier.qualify_regimes({
                    'regime_labels': regime_labels,
                    'regime_centers': regime_centers,
                    'regime_statistics': regime_statistics
                }, data)
                qualified_regimes = qualification_result.get('qualified_regimes', {})
            
            # Analyze regimes
            regime_analysis = self._analyze_regimes(regime_labels, regime_centers, regime_statistics, data)
            
            # Generate regime features
            regime_features = self._generate_regime_features(data, regime_labels, regime_centers)
            
            # Calculate regime quality scores
            regime_quality_scores = self._calculate_regime_quality_scores(regime_labels, regime_centers, regime_statistics)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(regime_data, regime_labels)
            
            # Calculate detection time
            detection_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive result
            result = RegimeResult(
                # Regime data
                regime_labels=regime_labels,
                regime_centers=regime_centers,
                regime_statistics=regime_statistics,
                qualified_regimes=qualified_regimes,
                
                # Regime analysis
                regime_transitions=regime_analysis['transitions'],
                regime_stability=regime_analysis['stability'],
                regime_persistence=regime_analysis['persistence'],
                regime_quality_scores=regime_quality_scores,
                
                # Regime features
                regime_features=regime_features,
                regime_feature_importance=regime_analysis['feature_importance'],
                regime_feature_correlations=regime_analysis['feature_correlations'],
                
                # Detection metadata
                detection_method=self.config.detection_method.value,
                detection_time=detection_time,
                n_regimes_detected=len(np.unique(regime_labels)),
                n_qualified_regimes=len(qualified_regimes),
                regime_quality_score=np.mean(list(regime_quality_scores.values())),
                
                # Performance metrics
                silhouette_score=performance_metrics['silhouette_score'],
                calinski_harabasz_score=performance_metrics['calinski_harabasz_score'],
                davies_bouldin_score=performance_metrics['davies_bouldin_score'],
                
                # Metadata
                config=self.config
            )
            
            # Save regime data if configured
            if self.config.save_regime_data:
                self._save_regime_data(result)
            
            self.logger.info(f"✅ Regime detection completed in {result.detection_time:.2f}s")
            self.logger.info(f"📊 Regimes detected: {result.n_regimes_detected}")
            self.logger.info(f"📊 Qualified regimes: {result.n_qualified_regimes}")
            self.logger.info(f"📊 Regime quality score: {result.regime_quality_score:.3f}")
            self.logger.info(f"📊 Silhouette score: {result.silhouette_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Regime detection failed: {e}")
            raise
    
    def _prepare_regime_data(self, data: pd.DataFrame, features: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Prepare data for regime detection."""
        try:
            if features is not None:
                # Use provided features
                regime_data = features.copy()
            else:
                # Generate basic features for regime detection
                regime_data = pd.DataFrame(index=data.index)
                
                # Price features
                if 'close' in data.columns:
                    regime_data['price_return'] = data['close'].pct_change()
                    regime_data['price_volatility'] = data['close'].rolling(window=20).std()
                    regime_data['price_trend'] = data['close'].rolling(window=20).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan
                    )
                
                # Volume features
                if 'volume' in data.columns:
                    regime_data['volume_return'] = data['volume'].pct_change()
                    regime_data['volume_volatility'] = data['volume'].rolling(window=20).std()
                
                # Technical indicators
                if 'close' in data.columns:
                    # RSI
                    delta = data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    regime_data['rsi'] = 100 - (100 / (1 + rs))
                    
                    # Moving averages
                    regime_data['sma_20'] = data['close'].rolling(window=20).mean()
                    regime_data['sma_50'] = data['close'].rolling(window=50).mean()
                    regime_data['sma_ratio'] = regime_data['sma_20'] / regime_data['sma_50']
            
            # Fill missing values
            regime_data = regime_data.fillna(regime_data.median())
            
            return regime_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime data preparation failed: {e}")
            return data
    
    def _detect_unsupervised_regimes(self, regime_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        """Detect regimes using unsupervised methods."""
        try:
            # Use the unsupervised regime detector
            regime_result = self.regime_detector.detect_regimes(regime_data)
            
            regime_labels = regime_result.get('regime_labels', np.array([]))
            regime_centers = regime_result.get('regime_centers', {})
            regime_statistics = regime_result.get('regime_statistics', {})
            
            return regime_labels, regime_centers, regime_statistics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Unsupervised regime detection failed: {e}")
            return self._detect_basic_regimes(regime_data)
    
    def _detect_basic_regimes(self, regime_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        """Detect regimes using basic methods."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data
            numeric_cols = regime_data.select_dtypes(include=[np.number]).columns
            regime_data_numeric = regime_data[numeric_cols].fillna(0)
            
            # Scale data
            scaler = StandardScaler()
            regime_data_scaled = scaler.fit_transform(regime_data_numeric)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
            regime_labels = kmeans.fit_predict(regime_data_scaled)
            
            # Calculate regime centers
            regime_centers = {}
            for i in range(self.config.n_regimes):
                regime_centers[i] = kmeans.cluster_centers_[i]
            
            # Calculate regime statistics
            regime_statistics = {}
            for i in range(self.config.n_regimes):
                regime_mask = regime_labels == i
                regime_data_subset = regime_data_numeric[regime_mask]
                
                regime_statistics[i] = {
                    'count': np.sum(regime_mask),
                    'percentage': np.sum(regime_mask) / len(regime_labels),
                    'mean': regime_data_subset.mean().to_dict(),
                    'std': regime_data_subset.std().to_dict(),
                    'min': regime_data_subset.min().to_dict(),
                    'max': regime_data_subset.max().to_dict()
                }
            
            return regime_labels, regime_centers, regime_statistics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Basic regime detection failed: {e}")
            # Fallback to simple regime assignment
            regime_labels = np.zeros(len(regime_data))
            regime_centers = {0: np.zeros(regime_data.shape[1])}
            regime_statistics = {0: {'count': len(regime_data), 'percentage': 1.0}}
            
            return regime_labels, regime_centers, regime_statistics
    
    def _analyze_regimes(self, regime_labels: np.ndarray, regime_centers: Dict[int, np.ndarray], 
                        regime_statistics: Dict[int, Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze detected regimes."""
        try:
            analysis = {
                'transitions': [],
                'stability': {},
                'persistence': {},
                'feature_importance': {},
                'feature_correlations': pd.DataFrame()
            }
            
            # Analyze regime transitions
            if self.config.regime_transitions:
                transitions = self._analyze_regime_transitions(regime_labels)
                analysis['transitions'] = transitions
            
            # Analyze regime stability
            if self.config.regime_stability:
                stability = self._analyze_regime_stability(regime_labels, regime_centers)
                analysis['stability'] = stability
            
            # Analyze regime persistence
            if self.config.regime_persistence:
                persistence = self._analyze_regime_persistence(regime_labels)
                analysis['persistence'] = persistence
            
            # Analyze feature importance
            feature_importance = self._analyze_feature_importance(regime_labels, data)
            analysis['feature_importance'] = feature_importance
            
            # Analyze feature correlations
            feature_correlations = self._analyze_feature_correlations(regime_labels, data)
            analysis['feature_correlations'] = feature_correlations
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime analysis failed: {e}")
            return {
                'transitions': [],
                'stability': {},
                'persistence': {},
                'feature_importance': {},
                'feature_correlations': pd.DataFrame()
            }
    
    def _analyze_regime_transitions(self, regime_labels: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze regime transitions."""
        try:
            transitions = []
            
            for i in range(1, len(regime_labels)):
                if regime_labels[i] != regime_labels[i-1]:
                    transitions.append({
                        'timestamp': i,
                        'from_regime': int(regime_labels[i-1]),
                        'to_regime': int(regime_labels[i]),
                        'transition_type': f"regime_{int(regime_labels[i-1])}_to_regime_{int(regime_labels[i])}"
                    })
            
            return transitions
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime transition analysis failed: {e}")
            return []
    
    def _analyze_regime_stability(self, regime_labels: np.ndarray, regime_centers: Dict[int, np.ndarray]) -> Dict[int, float]:
        """Analyze regime stability."""
        try:
            stability = {}
            
            for regime_id in np.unique(regime_labels):
                regime_mask = regime_labels == regime_id
                regime_duration = np.sum(regime_mask)
                total_duration = len(regime_labels)
                
                # Calculate stability as ratio of regime duration to total duration
                stability[regime_id] = regime_duration / total_duration
            
            return stability
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime stability analysis failed: {e}")
            return {}
    
    def _analyze_regime_persistence(self, regime_labels: np.ndarray) -> Dict[int, float]:
        """Analyze regime persistence."""
        try:
            persistence = {}
            
            for regime_id in np.unique(regime_labels):
                regime_mask = regime_labels == regime_id
                regime_indices = np.where(regime_mask)[0]
                
                if len(regime_indices) > 1:
                    # Calculate average gap between regime occurrences
                    gaps = np.diff(regime_indices)
                    persistence[regime_id] = np.mean(gaps) if len(gaps) > 0 else 0
                else:
                    persistence[regime_id] = 0
            
            return persistence
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime persistence analysis failed: {e}")
            return {}
    
    def _analyze_feature_importance(self, regime_labels: np.ndarray, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze feature importance for regime detection."""
        try:
            from sklearn.feature_selection import mutual_info_classification
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data_numeric = data[numeric_cols].fillna(0)
            
            # Calculate mutual information between features and regime labels
            importance_scores = mutual_info_classification(data_numeric, regime_labels)
            
            feature_importance = dict(zip(numeric_cols, importance_scores))
            
            return feature_importance
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature importance analysis failed: {e}")
            return {}
    
    def _analyze_feature_correlations(self, regime_labels: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze feature correlations within regimes."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data_numeric = data[numeric_cols].fillna(0)
            
            # Calculate correlations
            correlations = data_numeric.corr()
            
            return correlations
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature correlation analysis failed: {e}")
            return pd.DataFrame()
    
    def _generate_regime_features(self, data: pd.DataFrame, regime_labels: np.ndarray, 
                                regime_centers: Dict[int, np.ndarray]) -> pd.DataFrame:
        """Generate regime-specific features."""
        try:
            regime_features = pd.DataFrame(index=data.index)
            
            # Add regime labels
            regime_features['regime_label'] = regime_labels
            
            # Add regime-specific features
            for regime_id in np.unique(regime_labels):
                regime_mask = regime_labels == regime_id
                regime_features[f'regime_{regime_id}'] = regime_mask.astype(int)
            
            # Add regime distance features
            if len(regime_centers) > 0:
                for regime_id, center in regime_centers.items():
                    # Calculate distance to regime center (simplified)
                    regime_features[f'distance_to_regime_{regime_id}'] = 0.0  # Placeholder
            
            return regime_features
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"⚠️ Regime feature generation failed due to data type issue: {e}")
            self.logger.warning(f"Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}, regime_labels shape: {regime_labels.shape if hasattr(regime_labels, 'shape') else 'N/A'}")
            return pd.DataFrame(index=data.index)
        except (MemoryError, OSError) as e:
            self.logger.warning(f"⚠️ Regime feature generation failed due to system resource issue: {e}")
            return pd.DataFrame(index=data.index)
        except Exception as e:
            self.logger.warning(f"⚠️ Regime feature generation failed with unexpected error: {e}")
            self.logger.warning(f"Error type: {type(e).__name__}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_regime_quality_scores(self, regime_labels: np.ndarray, regime_centers: Dict[int, np.ndarray], 
                                            regime_statistics: Dict[int, Dict[str, Any]]) -> Dict[int, float]:
        """Calculate regime quality scores."""
        try:
            quality_scores = {}
            
            for regime_id in np.unique(regime_labels):
                regime_mask = regime_labels == regime_id
                regime_count = np.sum(regime_mask)
                total_count = len(regime_labels)
                
                # Calculate quality score based on regime size and statistics
                size_score = regime_count / total_count
                consistency_score = 1.0  # Placeholder for consistency calculation
                
                quality_scores[regime_id] = (size_score + consistency_score) / 2
            
            return quality_scores
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"⚠️ Regime quality score calculation failed due to data type issue: {e}")
            self.logger.warning(f"Regime labels shape: {regime_labels.shape if hasattr(regime_labels, 'shape') else 'N/A'}")
        except (MemoryError, OSError) as e:
            self.logger.warning(f"⚠️ Regime quality score calculation failed due to system resource issue: {e}")
        except Exception as e:
            self.logger.warning(f"⚠️ Regime quality score calculation failed with unexpected error: {e}")
            self.logger.warning(f"Error type: {type(e).__name__}")
            return {}
    
    def _calculate_performance_metrics(self, regime_data: pd.DataFrame, regime_labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering performance metrics."""
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            # Prepare data
            numeric_cols = regime_data.select_dtypes(include=[np.number]).columns
            regime_data_numeric = regime_data[numeric_cols].fillna(0)
            
            # Calculate metrics
            silhouette = silhouette_score(regime_data_numeric, regime_labels)
            calinski_harabasz = calinski_harabasz_score(regime_data_numeric, regime_labels)
            davies_bouldin = davies_bouldin_score(regime_data_numeric, regime_labels)
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance metrics calculation failed: {e}")
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': 0.0
            }
    
    def _save_regime_data(self, result: RegimeResult):
        """Save regime data to file."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save regime data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"regime_data_{timestamp}.parquet"
            filepath = output_dir / filename
            
            result.regime_features.to_parquet(filepath)
            
            # Save metadata
            metadata_file = output_dir / f"regime_metadata_{timestamp}.json"
            import json
            metadata = {
                'regime_labels': result.regime_labels.tolist(),
                'regime_centers': {str(k): v.tolist() for k, v in result.regime_centers.items()},
                'regime_statistics': result.regime_statistics,
                'qualified_regimes': result.qualified_regimes,
                'detection_method': result.detection_method,
                'detection_time': result.detection_time,
                'n_regimes_detected': result.n_regimes_detected,
                'n_qualified_regimes': result.n_qualified_regimes,
                'regime_quality_score': result.regime_quality_score,
                'silhouette_score': result.silhouette_score,
                'calinski_harabasz_score': result.calinski_harabasz_score,
                'davies_bouldin_score': result.davies_bouldin_score
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"📁 Regime data saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save regime data: {e}")
    
    def export_regime_data(self, result: RegimeResult, filepath: str):
        """Export regime data to file."""
        try:
            result.regime_features.to_csv(filepath)
            self.logger.info(f"📁 Regime data exported to {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Failed to export regime data: {e}")