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
    from ..regime_analysis.unsupervised_regime_detection import UnsupervisedRegimeDetector, RegimeDetectionConfig
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
        tprint_info("🔍 Initializing Regime Detection Pipeline")
        tprint_debug(f"Configuration: {config}")
        tprint_debug(f"Detection method: {config.detection_method.value}")
        tprint_debug(f"Number of regimes: {config.n_regimes}")
        tprint_debug(f"Qualification enabled: {config.enable_regime_qualification}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize performance tracking
        self.performance_metrics = {
            'data_preparation_time': 0.0,
            'regime_detection_time': 0.0,
            'evaluation_time': 0.0,
            'total_execution_time': 0.0
        }
        
        # Initialize regime detection components
        self.regime_detector = None
        self.regime_qualifier = None
        
        # Initialize available components
        tprint_debug("Initializing regime detection components...")
        self._initialize_components()
        
        self.logger.info("✅ Regime Detector initialized")
        self.logger.info(f"📊 Detection method: {config.detection_method.value}")
        self.logger.info(f"📊 Number of regimes: {config.n_regimes}")
        self.logger.info(f"📊 Qualification enabled: {config.enable_regime_qualification}")
        tprint_success("✅ Regime Detection Pipeline initialized successfully")
    
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
        tprint_info("🚀 Starting regime detection")
        tprint_debug(f"Input data shape: {data.shape}")
        tprint_debug(f"Features provided: {features is not None}")
        tprint_debug(f"Detection method: {self.config.detection_method.value}")
        tprint_debug(f"Number of regimes: {self.config.n_regimes}")
        
        self.logger.info("🚀 Starting regime detection")
        start_time = datetime.now()
        
        try:
            # Prepare data for regime detection
            tprint_debug("Preparing data for regime detection...")
            regime_data = self._prepare_regime_data(data, features)
            tprint_debug(f"Regime data shape: {regime_data.shape}")
            tprint_debug(f"Regime data type: {regime_data.dtype}")
            
            # Detect regimes
            tprint_debug(f"Using detection method: {self.config.detection_method.value}")
            if self.config.detection_method == RegimeDetectionMethod.UNSUPERVISED and self.regime_detector:
                tprint_debug("Using unsupervised regime detection...")
                regime_labels, regime_centers, regime_statistics = self._detect_unsupervised_regimes(regime_data)
            else:
                tprint_debug("Using basic regime detection...")
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
        tprint_debug("Preparing data for regime detection...")
        tprint_debug(f"Data shape: {data.shape}")
        tprint_debug(f"Features provided: {features is not None}")
        if features is not None:
            tprint_debug(f"Features shape: {features.shape}")
        
        prep_start = time.time()
        
        try:
            if features is not None:
                # Use provided features
                tprint_debug("Using provided features...")
                regime_data = features.copy()
                tprint_debug(f"Using features shape: {regime_data.shape}")
            else:
                # Generate basic features for regime detection
                tprint_debug("Generating basic features for regime detection...")
                regime_data = pd.DataFrame(index=data.index)
                
                # Price features
                if 'close' in data.columns:
                    tprint_debug("Generating price features...")
                    regime_data['price_return'] = data['close'].pct_change()
                    regime_data['price_volatility'] = rolling_std(data["close"], window=20) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=20).std()
                    regime_data['price_trend'] = data['close'].rolling(window=20).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan
                    )
                    tprint_debug("Price features generated")
                
                # Volume features
                if 'volume' in data.columns:
                    tprint_debug("Generating volume features...")
                    regime_data['volume_return'] = data['volume'].pct_change()
                    regime_data['volume_volatility'] = rolling_std(data["volume"], window=20) if VECTORBT_AVAILABLE and len(data) > 1000 else data["volume"].rolling(window=20).std()
                    tprint_debug("Volume features generated")
                
                # Technical indicators
                if 'close' in data.columns:
                    tprint_debug("Generating technical indicators...")
                    # RSI
                    delta = data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    regime_data['rsi'] = 100 - (100 / (1 + rs))
                    
                    # Moving averages
                    regime_data['sma_20'] = rolling_mean(data["close"], window=20) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=20).mean()
                    regime_data['sma_50'] = rolling_mean(data["close"], window=50) if VECTORBT_AVAILABLE and len(data) > 1000 else data["close"].rolling(window=50).mean()
                    regime_data['sma_ratio'] = regime_data['sma_20'] / regime_data['sma_50']
                    
                    tprint_debug("Technical indicators generated")
            
            # Fill missing values
            tprint_debug("Filling missing values...")
            regime_data = regime_data.fillna(regime_data.median())
            tprint_debug(f"Data after filling missing values: {regime_data.shape}")
            tprint_debug(f"Missing values: {regime_data.isnull().sum().sum()}")
            
            prep_time = time.time() - prep_start
            
            tprint_debug(f"Data preparation completed in {prep_time:.3f}s")
            tprint_debug(f"Final regime data shape: {regime_data.shape}")
            tprint_debug(f"Regime data columns: {list(regime_data.columns)}")
            
            return regime_data
            
        except Exception as e:
            prep_time = time.time() - prep_start
            self.logger.warning(f"⚠️ Regime data preparation failed: {e}")
            tprint_error(f"❌ Regime data preparation failed: {e}", color="red")
            tprint_error(f"   Preparation time before failure: {prep_time:.3f}s", color="red")
            tprint_error(f"   Returning original data", color="red")
            
            return data
    
    def _detect_unsupervised_regimes(self, regime_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        """Detect regimes using unsupervised methods."""
        tprint_debug("Detecting regimes using unsupervised methods...")
        tprint_debug(f"Regime data shape: {regime_data.shape}")
        
        detection_start = time.time()
        
        try:
            # Use the unsupervised regime detector
            regime_result = self.regime_detector.detect_regimes(regime_data)
            
            regime_labels = regime_result.get('regime_labels', np.array([]))
            regime_centers = regime_result.get('regime_centers', {})
            regime_statistics = regime_result.get('regime_statistics', {})
            
            detection_time = time.time() - detection_start
            
            tprint_debug(f"Unsupervised regime detection completed in {detection_time:.3f}s")
            tprint_debug(f"Regime labels shape: {regime_labels.shape}")
            tprint_debug(f"Regime centers: {len(regime_centers)}")
            tprint_debug(f"Regime statistics: {len(regime_statistics)}")
            
            # Log detailed regime information
            for regime_id, stats in regime_statistics.items():
                tprint_debug(f"Regime {regime_id}: {stats}")
            
            return regime_labels, regime_centers, regime_statistics
            
        except Exception as e:
            detection_time = time.time() - detection_start
            self.logger.warning(f"⚠️ Unsupervised regime detection failed: {e}")
            tprint_error(f"❌ Unsupervised regime detection failed: {e}", color="red")
            tprint_error(f"   Detection time before failure: {detection_time:.3f}s", color="red")
            tprint_error(f"   Falling back to basic regime detection", color="red")
            
            # Fallback to basic regime detection
            tprint_debug("Attempting fallback to basic regime detection...")
            return self._detect_basic_regimes(regime_data)
    
    def _detect_basic_regimes(self, regime_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        """Detect regimes using basic methods."""
        tprint_debug("Detecting basic regimes using sequential assignment")
        tprint_debug(f"Regime data shape: {regime_data.shape}")
        tprint_debug(f"Number of regimes: {self.config.n_regimes}")
        
        detection_start = time.time()
        
        try:
            # KMeans clustering removed - will be handled in subsequent step
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data
            tprint_debug("Preparing data for regime detection...")
            numeric_cols = regime_data.select_dtypes(include=[np.number]).columns
            regime_data_numeric = regime_data[numeric_cols].fillna(0)
            
            tprint_debug(f"Numeric columns: {len(numeric_cols)}")
            tprint_debug(f"Data shape after preparation: {regime_data_numeric.shape}")
            
            # Scale data
            tprint_debug("Scaling data...")
            scaler = StandardScaler()
            regime_data_scaled = scaler.fit_transform(regime_data_numeric)
            
            tprint_debug(f"Scaled data shape: {regime_data_scaled.shape}")
            tprint_debug(f"Scaled data range: {np.min(regime_data_scaled):.3f} to {np.max(regime_data_scaled):.3f}")
            
            # Perform clustering
            # Simple regime assignment instead of KMeans
            n_samples = len(regime_data_numeric)
            regime_size = n_samples // self.config.n_regimes
            
            tprint_debug(f"Number of samples: {n_samples}")
            tprint_debug(f"Regime size: {regime_size}")
            
            regime_labels = np.array([i // regime_size for i in range(n_samples)])
            regime_labels = np.minimum(regime_labels, self.config.n_regimes - 1)
            
            tprint_debug(f"Regime labels shape: {regime_labels.shape}")
            tprint_debug(f"Unique regime labels: {len(np.unique(regime_labels))}")
            tprint_debug(f"Regime label distribution: {np.bincount(regime_labels)}")
            # regime_labels already assigned above
            
            # Calculate regime centers
            tprint_debug("Calculating regime centers...")
            regime_centers = {}
            for i in range(self.config.n_regimes):
                regime_mask = regime_labels == i
                if np.sum(regime_mask) > 0:
                    regime_centers[i] = np.mean(regime_data_scaled[regime_mask], axis=0)
                    tprint_debug(f"Regime {i} center: {regime_centers[i]}")
                else:
                    regime_centers[i] = np.zeros(regime_data_scaled.shape[1])
                    tprint_debug(f"Regime {i} center (zero): {regime_centers[i]}")
            
            tprint_debug(f"Regime centers calculated for {len(regime_centers)} regimes")
            
            # Calculate regime statistics
            tprint_debug("Calculating regime statistics...")
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
                
                tprint_debug(f"Regime {i} statistics: count={regime_statistics[i]['count']}, percentage={regime_statistics[i]['percentage']:.3f}")
            
            detection_time = time.time() - detection_start
            
            tprint_debug(f"Basic regime detection completed in {detection_time:.3f}s")
            tprint_debug(f"Regime labels shape: {regime_labels.shape}")
            tprint_debug(f"Regime centers: {len(regime_centers)}")
            tprint_debug(f"Regime statistics: {len(regime_statistics)}")
            
            # Log detailed regime information
            for regime_id, stats in regime_statistics.items():
                tprint_debug(f"Regime {regime_id}: {stats}")
            
            return regime_labels, regime_centers, regime_statistics
            
        except Exception as e:
            detection_time = time.time() - detection_start
            self.logger.warning(f"⚠️ Basic regime detection failed: {e}")
            tprint_error(f"❌ Basic regime detection failed: {e}", color="red")
            tprint_error(f"   Detection time before failure: {detection_time:.3f}s", color="red")
            tprint_error(f"   Falling back to simple regime assignment", color="red")
            
            # Fallback to simple regime assignment
            regime_labels = np.zeros(len(regime_data))
            regime_centers = {0: np.zeros(regime_data.shape[1])}
            regime_statistics = {0: {'count': len(regime_data), 'percentage': 1.0}}
            
            tprint_debug(f"Fallback regime labels shape: {regime_labels.shape}")
            tprint_debug(f"Fallback regime centers: {len(regime_centers)}")
            tprint_debug(f"Fallback regime statistics: {regime_statistics}")
            
            tprint_debug("Fallback regime assignment completed")
            tprint_debug(f"All data points assigned to regime 0")
            tprint_debug(f"Fallback regime center shape: {regime_centers[0].shape}")
            
            return regime_labels, regime_centers, regime_statistics
    
    def _analyze_regimes(self, regime_labels: np.ndarray, regime_centers: Dict[int, np.ndarray], 
                        regime_statistics: Dict[int, Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze detected regimes."""
        tprint_debug("Analyzing detected regimes...")
        tprint_debug(f"Regime labels shape: {regime_labels.shape}")
        tprint_debug(f"Regime centers: {len(regime_centers)}")
        tprint_debug(f"Regime statistics: {len(regime_statistics)}")
        tprint_debug(f"Data shape: {data.shape}")
        
        analysis_start = time.time()
        
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
                tprint_debug("Analyzing regime transitions...")
                transitions = self._analyze_regime_transitions(regime_labels)
                analysis['transitions'] = transitions
                tprint_debug(f"Regime transitions: {len(transitions)}")
            
            # Analyze regime stability
            if self.config.regime_stability:
                tprint_debug("Analyzing regime stability...")
                stability = self._analyze_regime_stability(regime_labels, regime_centers)
                analysis['stability'] = stability
                tprint_debug(f"Regime stability: {stability}")
            
            # Analyze regime persistence
            if self.config.regime_persistence:
                tprint_debug("Analyzing regime persistence...")
                persistence = self._analyze_regime_persistence(regime_labels)
                analysis['persistence'] = persistence
                tprint_debug(f"Regime persistence: {persistence}")
            
            # Analyze feature importance
            tprint_debug("Analyzing feature importance...")
            feature_importance = self._analyze_feature_importance(regime_labels, data)
            analysis['feature_importance'] = feature_importance
            tprint_debug(f"Feature importance: {feature_importance}")
            
            # Analyze feature correlations
            tprint_debug("Analyzing feature correlations...")
            feature_correlations = self._analyze_feature_correlations(regime_labels, data)
            analysis['feature_correlations'] = feature_correlations
            tprint_debug(f"Feature correlations shape: {feature_correlations.shape}")
            
            analysis_time = time.time() - analysis_start
            
            tprint_debug(f"Regime analysis completed in {analysis_time:.3f}s")
            tprint_debug(f"Analysis results: {list(analysis.keys())}")
            
            # Log detailed analysis results
            for key, value in analysis.items():
                if isinstance(value, (int, float)):
                    tprint_debug(f"Analysis {key}: {value}")
                elif isinstance(value, dict):
                    tprint_debug(f"Analysis {key}: {len(value)} items")
                else:
                    tprint_debug(f"Analysis {key}: {type(value)}")
            
            return analysis
            
        except Exception as e:
            analysis_time = time.time() - analysis_start
            self.logger.warning(f"⚠️ Regime analysis failed: {e}")
            tprint_error(f"❌ Regime analysis failed: {e}", color="red")
            tprint_error(f"   Analysis time before failure: {analysis_time:.3f}s", color="red")
            tprint_error(f"   Returning empty analysis results", color="red")
            
            tprint_debug("Returning empty analysis results")
            tprint_debug("This may indicate issues with regime detection or data quality")
            
            return {
                'transitions': [],
                'stability': {},
                'persistence': {},
                'feature_importance': {},
                'feature_correlations': pd.DataFrame()
            }
    
    def _analyze_regime_transitions(self, regime_labels: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze regime transitions."""
        tprint_debug("Analyzing regime transitions...")
        tprint_debug(f"Regime labels shape: {regime_labels.shape}")
        
        transition_start = time.time()
        
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
            
            transition_time = time.time() - transition_start
            
            tprint_debug(f"Regime transition analysis completed in {transition_time:.3f}s")
            tprint_debug(f"Number of transitions: {len(transitions)}")
            tprint_debug(f"Transitions: {transitions}")
            
            transition_time = time.time() - transition_start
            
            tprint_debug(f"Regime transition analysis completed in {transition_time:.3f}s")
            tprint_debug(f"Number of transitions: {len(transitions)}")
            tprint_debug(f"Transitions: {transitions}")
            
            return transitions
            
        except Exception as e:
            transition_time = time.time() - transition_start
            self.logger.warning(f"⚠️ Regime transition analysis failed: {e}")
            tprint_error(f"❌ Regime transition analysis failed: {e}", color="red")
            tprint_error(f"   Transition analysis time before failure: {transition_time:.3f}s", color="red")
            tprint_error(f"   Returning empty transitions list", color="red")
            
            tprint_debug("Returning empty transitions list")
            tprint_debug("This may indicate issues with regime detection or data quality")
            
            return []
    
    def _analyze_regime_stability(self, regime_labels: np.ndarray, regime_centers: Dict[int, np.ndarray]) -> Dict[int, float]:
        """Analyze regime stability."""
        tprint_debug("Analyzing regime stability...")
        tprint_debug(f"Regime labels shape: {regime_labels.shape}")
        tprint_debug(f"Regime centers: {len(regime_centers)}")
        
        stability_start = time.time()
        
        try:
            stability = {}
            
            for regime_id in np.unique(regime_labels):
                regime_mask = regime_labels == regime_id
                regime_duration = np.sum(regime_mask)
                total_duration = len(regime_labels)
                
                # Calculate stability as ratio of regime duration to total duration
                stability[regime_id] = regime_duration / total_duration
                tprint_debug(f"Regime {regime_id} stability: {stability[regime_id]:.3f} (duration: {regime_duration}/{total_duration})")
            
            stability_time = time.time() - stability_start
            
            tprint_debug(f"Regime stability analysis completed in {stability_time:.3f}s")
            tprint_debug(f"Stability results: {stability}")
            
            # Log detailed stability information
            for regime_id, score in stability.items():
                tprint_debug(f"Regime {regime_id} stability: {score:.4f}")
            
            return stability
            
        except Exception as e:
            stability_time = time.time() - stability_start
            self.logger.warning(f"⚠️ Regime stability analysis failed: {e}")
            tprint_error(f"❌ Regime stability analysis failed: {e}", color="red")
            tprint_error(f"   Stability analysis time before failure: {stability_time:.3f}s", color="red")
            tprint_error(f"   Returning empty stability results", color="red")
            
            tprint_debug("Returning empty stability dictionary")
            tprint_debug("This may indicate issues with regime detection or data quality")
            
            return {}
    
    def _analyze_regime_persistence(self, regime_labels: np.ndarray) -> Dict[int, float]:
        """Analyze regime persistence."""
        tprint_debug("Analyzing regime persistence...")
        tprint_debug(f"Regime labels shape: {regime_labels.shape}")
        
        persistence_start = time.time()
        
        try:
            persistence = {}
            
            for regime_id in np.unique(regime_labels):
                regime_mask = regime_labels == regime_id
                regime_indices = np.where(regime_mask)[0]
                
                if len(regime_indices) > 1:
                    # Calculate average gap between regime occurrences
                    gaps = np.diff(regime_indices)
                    persistence[regime_id] = np.mean(gaps) if len(gaps) > 0 else 0
                    tprint_debug(f"Regime {regime_id} persistence: {persistence[regime_id]:.3f} (gaps: {len(gaps)})")
                else:
                    persistence[regime_id] = 0
                    tprint_debug(f"Regime {regime_id} persistence: 0.0 (single occurrence)")
            
            persistence_time = time.time() - persistence_start
            
            tprint_debug(f"Regime persistence analysis completed in {persistence_time:.3f}s")
            tprint_debug(f"Persistence results: {persistence}")
            
            # Log detailed persistence information
            for regime_id, score in persistence.items():
                tprint_debug(f"Regime {regime_id} persistence: {score:.4f}")
            
            return persistence
            
        except Exception as e:
            persistence_time = time.time() - persistence_start
            self.logger.warning(f"⚠️ Regime persistence analysis failed: {e}")
            tprint_error(f"❌ Regime persistence analysis failed: {e}", color="red")
            tprint_error(f"   Persistence analysis time before failure: {persistence_time:.3f}s", color="red")
            tprint_error(f"   Returning empty persistence results", color="red")
            
            tprint_debug("Returning empty persistence dictionary")
            tprint_debug("This may indicate issues with regime detection or data quality")
            
            return {}
    
    def _analyze_feature_importance(self, regime_labels: np.ndarray, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze feature importance for regime detection."""
        tprint_debug("Analyzing feature importance...")
        tprint_debug(f"Regime labels shape: {regime_labels.shape}")
        tprint_debug(f"Data shape: {data.shape}")
        
        importance_start = time.time()
        
        try:
            from sklearn.feature_selection import mutual_info_classification
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data_numeric = data[numeric_cols].fillna(0)
            
            # Calculate mutual information between features and regime labels
            tprint_debug("Calculating mutual information between features and regime labels...")
            importance_scores = mutual_info_classification(data_numeric, regime_labels)
            
            tprint_debug(f"Importance scores shape: {importance_scores.shape}")
            tprint_debug(f"Importance scores range: {np.min(importance_scores):.3f} to {np.max(importance_scores):.3f}")
            
            feature_importance = dict(zip(numeric_cols, importance_scores))
            
            importance_time = time.time() - importance_start
            
            tprint_debug(f"Feature importance analysis completed in {importance_time:.3f}s")
            tprint_debug(f"Feature importance: {feature_importance}")
            
            # Log detailed feature importance information
            for feature, importance in feature_importance.items():
                tprint_debug(f"Feature {feature} importance: {importance:.4f}")
            
            return feature_importance
            
        except Exception as e:
            importance_time = time.time() - importance_start
            self.logger.warning(f"⚠️ Feature importance analysis failed: {e}")
            tprint_error(f"❌ Feature importance analysis failed: {e}", color="red")
            tprint_error(f"   Importance analysis time before failure: {importance_time:.3f}s", color="red")
            tprint_error(f"   Returning empty importance results", color="red")
            
            tprint_debug("Returning empty feature importance dictionary")
            tprint_debug("This may indicate issues with regime detection or data quality")
            
            return {}
    
    def _analyze_feature_correlations(self, regime_labels: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze feature correlations within regimes."""
        tprint_debug("Analyzing feature correlations...")
        tprint_debug(f"Regime labels shape: {regime_labels.shape}")
        tprint_debug(f"Data shape: {data.shape}")
        
        correlation_start = time.time()
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data_numeric = data[numeric_cols].fillna(0)
            
            tprint_debug(f"Numeric columns: {len(numeric_cols)}")
            tprint_debug(f"Data numeric shape: {data_numeric.shape}")
            
            # Calculate correlations
            tprint_debug("Calculating correlation matrix...")
            correlations = data_numeric.corr()
            
            correlation_time = time.time() - correlation_start
            
            tprint_debug(f"Feature correlation analysis completed in {correlation_time:.3f}s")
            tprint_debug(f"Correlation matrix shape: {correlations.shape}")
            tprint_debug(f"Correlation range: {correlations.min().min():.3f} to {correlations.max().max():.3f}")
            
            # Log detailed correlation information
            tprint_debug(f"Correlation matrix columns: {list(correlations.columns)}")
            tprint_debug(f"Correlation matrix index: {list(correlations.index)}")
            
            return correlations
            
        except Exception as e:
            correlation_time = time.time() - correlation_start
            self.logger.warning(f"⚠️ Feature correlation analysis failed: {e}")
            tprint_error(f"❌ Feature correlation analysis failed: {e}", color="red")
            tprint_error(f"   Correlation analysis time before failure: {correlation_time:.3f}s", color="red")
            tprint_error(f"   Returning empty correlation matrix", color="red")
            
            tprint_debug("Returning empty correlation matrix")
            tprint_debug("This may indicate issues with regime detection or data quality")
            
            return pd.DataFrame()
    
    def _generate_regime_features(self, data: pd.DataFrame, regime_labels: np.ndarray, 
                                regime_centers: Dict[int, np.ndarray]) -> pd.DataFrame:
        """Generate regime-specific features."""
        tprint_debug("Generating regime-specific features...")
        tprint_debug(f"Data shape: {data.shape}")
        tprint_debug(f"Regime labels shape: {regime_labels.shape}")
        tprint_debug(f"Regime centers: {len(regime_centers)}")
        
        feature_start = time.time()
        
        try:
            regime_features = pd.DataFrame(index=data.index)
            
            # Add regime labels
            regime_features['regime_label'] = regime_labels
            tprint_debug("Added regime labels as features")
            
            # Add regime-specific features
            for regime_id in np.unique(regime_labels):
                regime_mask = regime_labels == regime_id
                regime_features[f'regime_{regime_id}'] = regime_mask.astype(int)
                tprint_debug(f"Added regime {regime_id} indicator features")
            
            # Add regime distance features
            if len(regime_centers) > 0:
                for regime_id, center in regime_centers.items():
                    # Calculate distance to regime center (simplified)
                    regime_features[f'distance_to_regime_{regime_id}'] = 0.0  # Placeholder
                    tprint_debug(f"Added regime {regime_id} distance features (placeholder)")
            
            feature_time = time.time() - feature_start
            
            tprint_debug(f"Regime feature generation completed in {feature_time:.3f}s")
            tprint_debug(f"Final feature shape: {regime_features.shape}")
            tprint_debug(f"Feature columns: {list(regime_features.columns)}")
            
            # Log detailed feature information
            tprint_debug(f"Generated features data types: {regime_features.dtypes.to_dict()}")
            
            return regime_features
            
        except Exception as e:
            feature_time = time.time() - feature_start
            self.logger.warning(f"⚠️ Regime feature generation failed: {e}")
            tprint_error(f"❌ Regime feature generation failed: {e}", color="red")
            tprint_error(f"   Feature generation time before failure: {feature_time:.3f}s", color="red")
            tprint_error(f"   Returning empty DataFrame", color="red")
            
            tprint_debug("Returning empty DataFrame as fallback")
            tprint_debug("This may indicate issues with regime detection or data quality")
            
            return pd.DataFrame(index=data.index)
    
    def _calculate_regime_quality_scores(self, regime_labels: np.ndarray, regime_centers: Dict[int, np.ndarray], 
                                            regime_statistics: Dict[int, Dict[str, Any]]) -> Dict[int, float]:
        """Calculate regime quality scores."""
        tprint_debug("Calculating regime quality scores...")
        tprint_debug(f"Regime labels shape: {regime_labels.shape}")
        tprint_debug(f"Regime centers: {len(regime_centers)}")
        tprint_debug(f"Regime statistics: {len(regime_statistics)}")
        
        quality_start = time.time()
        
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
                tprint_debug(f"Regime {regime_id} quality score: {quality_scores[regime_id]:.3f} (size: {regime_count}/{total_count})")
            
            quality_time = time.time() - quality_start
            
            tprint_debug(f"Regime quality score calculation completed in {quality_time:.3f}s")
            tprint_debug(f"Quality scores: {quality_scores}")
            
            # Log detailed quality information
            for regime_id, score in quality_scores.items():
                tprint_debug(f"Regime {regime_id} quality: {score:.4f}")
            
            return quality_scores
            
        except Exception as e:
            quality_time = time.time() - quality_start
            self.logger.warning(f"⚠️ Regime quality score calculation failed: {e}")
            tprint_error(f"❌ Regime quality score calculation failed: {e}", color="red")
            tprint_error(f"   Quality calculation time before failure: {quality_time:.3f}s", color="red")
            tprint_error(f"   Returning empty quality scores", color="red")
            
            tprint_debug("Returning empty quality scores dictionary")
            tprint_debug("This may indicate issues with regime detection or data quality")
            
            return {}
    
    def _calculate_performance_metrics(self, regime_data: pd.DataFrame, regime_labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering performance metrics."""
        tprint_debug("Calculating performance metrics...")
        tprint_debug(f"Regime data shape: {regime_data.shape}")
        tprint_debug(f"Regime labels shape: {regime_labels.shape}")
        
        metrics_start = time.time()
        
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            # Prepare data
            numeric_cols = regime_data.select_dtypes(include=[np.number]).columns
            regime_data_numeric = regime_data[numeric_cols].fillna(0)
            
            # Calculate metrics
            tprint_debug("Calculating silhouette score...")
            silhouette = silhouette_score(regime_data_numeric, regime_labels)
            tprint_debug(f"Silhouette score: {silhouette:.3f}")
            
            tprint_debug("Calculating Calinski-Harabasz score...")
            calinski_harabasz = calinski_harabasz_score(regime_data_numeric, regime_labels)
            tprint_debug(f"Calinski-Harabasz score: {calinski_harabasz:.3f}")
            
            tprint_debug("Calculating Davies-Bouldin score...")
            davies_bouldin = davies_bouldin_score(regime_data_numeric, regime_labels)
            tprint_debug(f"Davies-Bouldin score: {davies_bouldin:.3f}")
            
            metrics_time = time.time() - metrics_start
            
            tprint_debug(f"Performance metrics calculation completed in {metrics_time:.3f}s")
            tprint_debug(f"Metrics: silhouette={silhouette:.3f}, calinski={calinski_harabasz:.3f}, davies={davies_bouldin:.3f}")
            
            metrics_time = time.time() - metrics_start
            
            tprint_debug(f"Performance metrics calculation completed in {metrics_time:.3f}s")
            tprint_debug(f"Metrics: silhouette={silhouette:.3f}, calinski={calinski_harabasz:.3f}, davies={davies_bouldin:.3f}")
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin
            }
            
        except Exception as e:
            metrics_time = time.time() - metrics_start
            self.logger.warning(f"⚠️ Performance metrics calculation failed: {e}")
            tprint_error(f"❌ Performance metrics calculation failed: {e}", color="red")
            tprint_error(f"   Metrics calculation time before failure: {metrics_time:.3f}s", color="red")
            tprint_error(f"   Returning zero metrics", color="red")
            
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': 0.0
            }
    
    def _save_regime_data(self, result: RegimeResult):
        """Save regime data to file."""
        tprint_debug("Saving regime data to file...")
        tprint_debug(f"Output directory: {self.config.output_directory}")
        
        save_start = time.time()
        
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            tprint_debug(f"Output directory created: {output_dir}")
            
            # Save regime data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"regime_data_{timestamp}.parquet"
            filepath = output_dir / filename
            
            tprint_debug(f"Saving regime features to {filepath}")
            result.regime_features.to_parquet(filepath)
            tprint_debug(f"Regime features saved: {filepath.stat().st_size / 1024:.2f} KB")
            
            # Save metadata
            metadata_file = output_dir / f"regime_metadata_{timestamp}.json"
            import json

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

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
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
            
            save_time = time.time() - save_start
            
            save_time = time.time() - save_start
            
            self.logger.info(f"📁 Regime data saved to {filepath}")
            tprint_success(f"✅ Regime data saved to {filepath}")
            tprint_debug(f"Save time: {save_time:.3f}s")
            tprint_debug(f"Metadata file: {metadata_file}")
            tprint_debug(f"Metadata file size: {metadata_file.stat().st_size / 1024:.2f} KB")
            
        except Exception as e:
            save_time = time.time() - save_start
            self.logger.warning(f"⚠️ Failed to save regime data: {e}")
            tprint_error(f"❌ Failed to save regime data: {e}", color="red")
            tprint_error(f"   Save time before failure: {save_time:.3f}s", color="red")
    
    def export_regime_data(self, result: RegimeResult, filepath: str):
        """Export regime data to file."""
        tprint_debug(f"Exporting regime data to {filepath}")
        tprint_debug(f"Regime features shape: {result.regime_features.shape}")
        
        export_start = time.time()
        
        try:
            result.regime_features.to_csv(filepath)
            
            export_time = time.time() - export_start
            
            export_time = time.time() - export_start
            
            self.logger.info(f"📁 Regime data exported to {filepath}")
            tprint_success(f"✅ Regime data exported to {filepath}")
            tprint_debug(f"Export time: {export_time:.3f}s")
            tprint_debug(f"File size: {Path(filepath).stat().st_size / 1024:.2f} KB")
            
        except Exception as e:
            export_time = time.time() - export_start
            self.logger.error(f"❌ Failed to export regime data: {e}")
            tprint_error(f"❌ Failed to export regime data: {e}", color="red")
            tprint_error(f"   Export time before failure: {export_time:.3f}s", color="red")

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
