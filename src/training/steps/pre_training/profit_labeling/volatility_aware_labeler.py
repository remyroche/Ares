"""
Volatility-Aware Multi-Horizon Profit Labeler

This module implements the main volatility-aware labeling system that explicitly accounts
for volatility and microstructure noise, optimized for creating strong labels that are
learnable by ML models and generalize well.

Key Features:
- Volatility-normalized target bands and horizons
- Event-based bar construction with microstructure filtering
- Noise gating and eligibility filters
- Multi-target scheme with data-driven selection
- Label quality scoring and optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import warnings

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_correlation
)
from src.utils.math_validation import MathValidation
from src.utils.serialization_utils import UniversalSerializer

# Import data leakage prevention
from src.utils.lookahead_bias_detector import LookaheadBiasDetector, get_global_detector

# Import data utilities
try:
    from src.utils.data.klines_parquet import KlineParquetManager
    from src.utils.data.unified_data_utils import UnifiedDataUtils
    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False

# Import components
from .bar_construction import EventBasedBarConstructor, BarConstructionConfig
from .volatility_modeling import VolatilityModeler, VolatilityConfig
from .noise_gating import NoiseGatingFilter, NoiseGatingConfig
from .quality_scoring import LabelQualityScorer, QualityScoringConfig
from .multi_target_scheme import MultiTargetScheme, MultiTargetConfig
from .enhanced_label_definitions import (
    EnhancedLabelDefinitions, LabelDefinitionType,
    AnalystLabelConfig, TacticianLabelConfig, RegimeConditionedConfig,
    RiskAwareConfig, DataCleaningConfig, StabilityCheckConfig,
    create_trading_aware_config
)


class LabelType(Enum):
    """Enumeration of label types."""
    HARD = "hard"  # -1, 0, +1
    SOFT = "soft"  # Confidence scores [0, 1]
    PROBABILITY = "probability"  # Probability distributions


@dataclass
class VolatilityAwareConfig:
    """Configuration for volatility-aware multi-horizon labeling."""
    
    # Bar construction settings
    bar_construction: BarConstructionConfig = field(default_factory=BarConstructionConfig)
    
    # Volatility modeling settings
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    
    # Noise gating settings
    noise_gating: NoiseGatingConfig = field(default_factory=NoiseGatingConfig)
    
    # Quality scoring settings
    quality_scoring: QualityScoringConfig = field(default_factory=QualityScoringConfig)
    
    # Multi-target scheme settings
    multi_target: MultiTargetConfig = field(default_factory=MultiTargetConfig)

    # Enhanced label definitions settings
    enable_enhanced_labels: bool = True
    label_definition_type: LabelDefinitionType = LabelDefinitionType.ANALYST

    # Enhanced label configurations
    analyst_config: AnalystLabelConfig = field(default_factory=AnalystLabelConfig)
    tactician_config: TacticianLabelConfig = field(default_factory=TacticianLabelConfig)
    regime_config: RegimeConditionedConfig = field(default_factory=RegimeConditionedConfig)
    risk_config: RiskAwareConfig = field(default_factory=RiskAwareConfig)
    cleaning_config: DataCleaningConfig = field(default_factory=DataCleaningConfig)
    stability_config: StabilityCheckConfig = field(default_factory=StabilityCheckConfig)

    # General settings
    min_data_points: int = 1000
    enable_caching: bool = True
    cache_duration_minutes: int = 60
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    
    # Output settings
    save_intermediate_results: bool = True
    output_directory: str = "volatility_aware_labeling_results"
    generate_reports: bool = True
    
    # Label quality thresholds
    min_auc_threshold: float = 0.55
    max_auc_std_threshold: float = 0.03
    min_psi_threshold: float = 0.1
    max_flip_rate_threshold: float = 0.15
    min_balance_threshold: float = 0.35
    max_balance_threshold: float = 0.65
    max_correlation_threshold: float = 0.4

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate data requirements
        if self.min_data_points < 100:
            raise ValueError("min_data_points must be at least 100")

        if self.cache_duration_minutes < 1:
            raise ValueError("cache_duration_minutes must be at least 1")

        # Validate thresholds
        if not (0 < self.min_auc_threshold < 1):
            raise ValueError("min_auc_threshold must be between 0 and 1")

        if not (0 < self.max_auc_std_threshold < 1):
            raise ValueError("max_auc_std_threshold must be between 0 and 1")

        if not (0 <= self.min_balance_threshold <= self.max_balance_threshold <= 1):
            raise ValueError("balance thresholds must satisfy: 0 ≤ min_balance ≤ max_balance ≤ 1")

        if not (0 < self.max_correlation_threshold < 1):
            raise ValueError("max_correlation_threshold must be between 0 and 1")

        # Validate component configurations
        if self.bar_construction:
            self.bar_construction._validate_config()
        if self.volatility:
            self.volatility._validate_config()
        if self.noise_gating:
            self.noise_gating._validate_config()
        if self.quality_scoring:
            self.quality_scoring._validate_config()
        if self.multi_target:
            self.multi_target._validate_config()


@dataclass
class LabelQualityScore:
    """Label quality score container."""
    
    # Core quality metrics
    predictability: float = 0.0  # AUC/PR-AUC from baselines
    stability: float = 0.0  # Variance of AUC across folds, PSI
    consistency: float = 0.0  # Mutual information between labels
    balance: float = 0.0  # Class balance
    snr_proxy: float = 0.0  # |IC| between features and labels
    
    # Composite score
    overall_quality: float = 0.0
    
    # Detailed metrics
    auc_mean: float = 0.0
    auc_std: float = 0.0
    psi_score: float = 0.0
    flip_rate: float = 0.0
    class_balance: float = 0.0
    mutual_information: float = 0.0
    information_coefficient: float = 0.0
    
    # Metadata
    n_samples: int = 0
    n_features: int = 0
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LabelingResult:
    """Result container for volatility-aware labeling."""
    
    # Core labeling results
    labels: pd.DataFrame
    confidence_scores: pd.DataFrame
    eligibility_masks: pd.DataFrame
    
    # Quality scores
    quality_scores: Dict[str, LabelQualityScore]
    
    # Component results
    bar_construction_result: Optional[Any] = None
    volatility_result: Optional[Any] = None
    noise_gating_result: Optional[Any] = None
    multi_target_result: Optional[Any] = None
    
    # Metadata
    config_used: VolatilityAwareConfig = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Statistics
    n_samples: int = 0
    n_targets: int = 0
    n_horizons: int = 0
    label_distribution: Dict[str, Any] = field(default_factory=dict)


class VolatilityAwareMultiHorizonLabeler:
    """
    Volatility-Aware Multi-Horizon Profit Labeler
    
    This class implements a comprehensive labeling system that explicitly accounts for
    volatility and microstructure noise, optimized for creating strong labels that are
    learnable by ML models and generalize well.
    
    Key Features:
    1. **Volatility-Normalized Targets**: Uses σ-units instead of fixed percentages
    2. **Event-Based Bar Construction**: Reduces microstructure noise through better bar formation
    3. **Noise Gating**: Filters out labels when noise dominates signal
    4. **Multi-Target Scheme**: Data-driven selection of small/medium/high targets
    5. **Quality Scoring**: Comprehensive assessment of label quality
    6. **Adaptive Horizons**: Data-driven horizon selection based on first-passage time
    """
    
    def __init__(self, config: Optional[VolatilityAwareConfig] = None):
        """Initialize volatility-aware multi-horizon profit labeler."""
        self.config = config or VolatilityAwareConfig()
        self.logger = logging.getLogger('VolatilityAwareLabeler')
        
        # Initialize components
        self.bar_constructor = EventBasedBarConstructor(self.config.bar_construction)
        self.volatility_modeler = VolatilityModeler(self.config.volatility)
        self.noise_gating_filter = NoiseGatingFilter(self.config.noise_gating)
        self.quality_scorer = LabelQualityScorer(self.config.quality_scoring)
        self.multi_target_scheme = MultiTargetScheme(self.config.multi_target)

        # Initialize enhanced label definitions if enabled
        if self.config.enable_enhanced_labels:
            self.enhanced_labeler = EnhancedLabelDefinitions(
                analyst_config=self.config.analyst_config,
                tactician_config=self.config.tactician_config,
                regime_config=self.config.regime_config,
                risk_config=self.config.risk_config,
                cleaning_config=self.config.cleaning_config,
                stability_config=self.config.stability_config
            )
        else:
            self.enhanced_labeler = None
        
        # State tracking
        self.labeling_history: List[LabelingResult] = []
        self.cache: Dict[str, Any] = {}

        # Intermediate computation caches
        self.bar_cache: Dict[str, Any] = {}
        self.volatility_cache: Dict[str, Any] = {}
        self.noise_cache: Dict[str, Any] = {}
        self.target_cache: Dict[str, Any] = {}
        self.quality_cache: Dict[str, Any] = {}

        # Initialize data leakage prevention
        self.lookahead_detector = LookaheadBiasDetector(strict_mode=True)

        # Initialize data utilities
        if DATA_UTILS_AVAILABLE:
            self.kline_manager = KlineParquetManager()
            self.data_utils = UnifiedDataUtils()
            tprint_info("   → Data utilities: Available")
        else:
            self.kline_manager = None
            self.data_utils = None
            tprint_warning("   → Data utilities: Not available")

        tprint_success("🚀 Volatility-Aware Multi-Horizon Profit Labeler initialized")
        tprint_info(f"   → Min data points: {self.config.min_data_points}")
        tprint_info(f"   → Parallel processing: {self.config.parallel_processing}")
        tprint_info(f"   → Caching enabled: {self.config.enable_caching}")
        tprint_info(f"   → Data leakage protection: Enabled")
    
    def generate_labels(self, market_data: pd.DataFrame) -> LabelingResult:
        """
        Generate volatility-aware profit labels.
        
        Args:
            market_data: OHLCV market data with datetime index
            
        Returns:
            LabelingResult with comprehensive labeling and analysis
        """
        start_time = datetime.now()
        tprint_info("🔍 Generating volatility-aware profit labels")
        
        # Validate input data
        if not self._validate_input_data(market_data):
            return self._create_empty_result()

        # Check for data leakage and lookahead bias
        if self.lookahead_detector:
            try:
                # Set current timestamp for bias detection (use last data point)
                if not market_data.empty:
                    self.lookahead_detector.set_current_timestamp(market_data.index[-1])

                # Validate no future data in input
                market_data = self.lookahead_detector.validate_dataframe_timestamps(
                    market_data, timestamp_column='timestamp' if 'timestamp' in market_data.columns else None
                )
            except Exception as e:
                tprint_error(f"❌ Data leakage detected: {e}")
                return self._create_empty_result()

        # Check main cache first
        cache_key = self._generate_cache_key(market_data)
        if self.config.enable_caching and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if self._is_cache_valid(cached_result):
                tprint_info("📋 Using cached labeling result")
                return cached_result

        # Check intermediate caches for reuse opportunities
        data_hash = self._generate_data_hash(market_data)
        
        # Initialize result container
        result = LabelingResult(
            labels=pd.DataFrame(),
            confidence_scores=pd.DataFrame(),
            eligibility_masks=pd.DataFrame(),
            quality_scores={},
            config_used=self.config
        )
        
        try:
            # Step 1: Event-based bar construction (with caching)
            tprint_info("📊 Step 1: Constructing event-based bars")
            bar_cache_key = f"bars_{data_hash}_{hash(str(self.config.bar_construction))}"
            if self.config.enable_caching and bar_cache_key in self.bar_cache:
                bar_result = self.bar_cache[bar_cache_key]
                tprint_info("📋 Using cached bar construction")
            else:
                bar_result = self.bar_constructor.construct_bars(market_data)
                if self.config.enable_caching:
                    self.bar_cache[bar_cache_key] = bar_result

            result.bar_construction_result = bar_result

            if bar_result.cleaned_bars.empty:
                tprint_warning("⚠️ No valid bars constructed")
                return self._create_empty_result()
            
            # Step 2: Volatility modeling (with caching)
            tprint_info("📈 Step 2: Modeling volatility")
            vol_cache_key = f"volatility_{data_hash}_{hash(str(self.config.volatility))}"
            if self.config.enable_caching and vol_cache_key in self.volatility_cache:
                vol_result = self.volatility_cache[vol_cache_key]
                tprint_info("📋 Using cached volatility modeling")
            else:
                vol_result = self.volatility_modeler.model_volatility(bar_result.cleaned_bars)
                if self.config.enable_caching:
                    self.volatility_cache[vol_cache_key] = vol_result

            result.volatility_result = vol_result

            if vol_result.volatility_series.empty:
                tprint_warning("⚠️ No volatility estimates available")
                return self._create_empty_result()
            
            # Step 3: Noise gating (with caching)
            tprint_info("🔇 Step 3: Applying noise gating")
            noise_cache_key = f"noise_{data_hash}_{hash(str(self.config.noise_gating))}"
            if self.config.enable_caching and noise_cache_key in self.noise_cache:
                noise_result = self.noise_cache[noise_cache_key]
                tprint_info("📋 Using cached noise gating")
            else:
                noise_result = self.noise_gating_filter.filter_noise(
                    bar_result.cleaned_bars, vol_result.volatility_series
                )
                if self.config.enable_caching:
                    self.noise_cache[noise_cache_key] = noise_result

            result.noise_gating_result = noise_result
            
            # Step 4: Multi-target scheme (with caching)
            tprint_info("🎯 Step 4: Generating multi-target labels")
            target_cache_key = f"targets_{data_hash}_{hash(str(self.config.multi_target))}"
            if self.config.enable_caching and target_cache_key in self.target_cache:
                target_result = self.target_cache[target_cache_key]
                tprint_info("📋 Using cached target generation")
            else:
                # Use enhanced label definitions if enabled
                if self.config.enable_enhanced_labels and self.enhanced_labeler:
                    tprint_info("🚀 Using enhanced label definitions")
                    target_result = self._generate_enhanced_targets(
                        bar_result.cleaned_bars,
                        vol_result.volatility_series,
                        noise_result.eligibility_mask
                    )
                else:
                    target_result = self.multi_target_scheme.generate_targets(
                        bar_result.cleaned_bars,
                        vol_result.volatility_series,
                        noise_result.eligibility_mask
                    )

                if self.config.enable_caching:
                    self.target_cache[target_cache_key] = target_result

            result.multi_target_result = target_result

            if target_result.labels.empty:
                tprint_warning("⚠️ No valid targets generated")
                return self._create_empty_result()

            # Step 5: Quality scoring (with caching)
            tprint_info("📊 Step 5: Assessing label quality")
            quality_cache_key = f"quality_{data_hash}_{hash(str(self.config.quality_scoring))}"
            if self.config.enable_caching and quality_cache_key in self.quality_cache:
                quality_scores = self.quality_cache[quality_cache_key]
                tprint_info("📋 Using cached quality scoring")
            else:
                quality_scores = self.quality_scorer.assess_quality(
                    target_result.labels,
                    target_result.confidence_scores,
                    target_result.eligibility_masks,
                    bar_result.cleaned_bars
                )
                if self.config.enable_caching:
                    self.quality_cache[quality_cache_key] = quality_scores

            result.quality_scores = quality_scores
            
            # Step 6: Filter by quality thresholds
            tprint_info("🔍 Step 6: Filtering by quality thresholds")
            filtered_result = self._filter_by_quality_thresholds(
                target_result, quality_scores
            )
            
            # Update result with filtered data
            result.labels = filtered_result.labels
            result.confidence_scores = filtered_result.confidence_scores
            result.eligibility_masks = filtered_result.eligibility_masks
            
            # Calculate statistics
            result.n_samples = len(result.labels)
            result.n_targets = len([col for col in result.labels.columns if 'target' in col])
            result.n_horizons = len([col for col in result.labels.columns if 'horizon' in col])
            result.label_distribution = self._calculate_label_distribution(result.labels)
            
        except Exception as e:
            tprint_error(f"❌ Labeling failed: {e}")
            return self._create_empty_result()
        
        # Calculate processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()

        # Store in history and cache
        self.labeling_history.append(result)
        if self.config.enable_caching:
            self.cache[cache_key] = result

        # Clean up old history
        if len(self.labeling_history) > 100:
            self.labeling_history = self.labeling_history[-100:]

        # Clean up old cache entries (keep only recent ones)
        self._cleanup_caches()

        tprint_success("✅ Volatility-aware labeling completed")
        tprint_info(f"   → Processing time: {result.processing_time:.2f}s")
        tprint_info(f"   → Samples: {result.n_samples}")
        tprint_info(f"   → Targets: {result.n_targets}")
        tprint_info(f"   → Horizons: {result.n_horizons}")

        return result

    def _generate_enhanced_targets(self, market_data: pd.DataFrame, volatility_series: pd.Series,
                                 eligibility_mask: pd.Series) -> Any:
        """
        Generate targets using enhanced label definitions.

        Args:
            market_data: Cleaned market data
            volatility_series: Volatility estimates
            eligibility_mask: Noise gating eligibility mask

        Returns:
            Target result object compatible with existing pipeline
        """
        try:
            tprint_info("🎯 Generating targets with enhanced label definitions")

            # Apply eligibility mask to market data
            eligible_data = market_data[eligibility_mask].copy()
            eligible_volatility = volatility_series[eligibility_mask].copy()

            if eligible_data.empty:
                tprint_warning("⚠️ No eligible data for enhanced labeling")
                return self.multi_target_scheme.generate_targets(
                    market_data, volatility_series, eligibility_mask
                )

            # Get regime data if available (would need to be passed in)
            # For now, create a simple regime classification based on volatility
            regime_data = self._classify_regimes_from_volatility(eligible_volatility)

            # Generate labels based on selected definition type
            if self.config.label_definition_type == LabelDefinitionType.ANALYST:
                analyst_labels, confidence_scores = self.enhanced_labeler.generate_analyst_labels(
                    eligible_data, eligible_volatility, regime_data
                )

                # Create target result structure compatible with existing pipeline
                target_result = type('TargetResult', (), {})()
                target_result.labels = pd.DataFrame({
                    'analyst_target': analyst_labels,
                    'analyst_confidence': confidence_scores
                })
                target_result.confidence_scores = pd.DataFrame({
                    'analyst_confidence': confidence_scores
                })
                target_result.eligibility_masks = pd.DataFrame({
                    'analyst_eligible': pd.Series(True, index=analyst_labels.index)
                })

            elif self.config.label_definition_type == LabelDefinitionType.TACTICIAN:
                tactician_labels, magnitude_scores = self.enhanced_labeler.generate_tactician_labels(
                    eligible_data, eligible_volatility, regime_data
                )

                target_result = type('TargetResult', (), {})()
                target_result.labels = pd.DataFrame({
                    'tactician_target': tactician_labels,
                    'tactician_magnitude': magnitude_scores
                })
                target_result.confidence_scores = pd.DataFrame({
                    'tactician_magnitude': magnitude_scores
                })
                target_result.eligibility_masks = pd.DataFrame({
                    'tactician_eligible': pd.Series(True, index=tactician_labels.index)
                })

            else:
                # Fall back to standard multi-target scheme
                tprint_warning(f"⚠️ Unsupported label definition type: {self.config.label_definition_type}")
                return self.multi_target_scheme.generate_targets(
                    market_data, volatility_series, eligibility_mask
                )

            # Apply stability checks if enabled
            if self.enhanced_labeler:
                stability_results = self.enhanced_labeler.check_label_stability(
                    target_result.labels.iloc[:, 0],  # Check first target column
                    market_data=eligible_data
                )

                if not stability_results['is_stable']:
                    tprint_warning("⚠️ Label stability issues detected:")
                    for issue in stability_results['issues']:
                        tprint_warning(f"   → {issue}")

            tprint_success("✅ Enhanced targets generated successfully")
            return target_result

        except Exception as e:
            tprint_error(f"❌ Error generating enhanced targets: {e}")
            # Fall back to standard targets
            return self.multi_target_scheme.generate_targets(
                market_data, volatility_series, eligibility_mask
            )

    def _classify_regimes_from_volatility(self, volatility_series: pd.Series) -> pd.Series:
        """
        Classify regimes based on volatility levels.

        Args:
            volatility_series: Volatility estimates

        Returns:
            Regime classifications
        """
        try:
            # Simple regime classification based on volatility percentiles
            low_threshold = volatility_series.quantile(0.33)
            high_threshold = volatility_series.quantile(0.67)

            regimes = pd.Series('normal', index=volatility_series.index)

            regimes[volatility_series <= low_threshold] = 'low_vol'
            regimes[volatility_series >= high_threshold] = 'high_vol'

            return regimes

        except Exception as e:
            tprint_warning(f"⚠️ Error classifying regimes: {e}")
            return pd.Series('normal', index=volatility_series.index)
    
    def _validate_input_data(self, market_data: pd.DataFrame) -> bool:
        """Validate input market data."""
        try:
            # Check if DataFrame is empty
            if market_data.empty:
                tprint_warning("⚠️ Input data is empty")
                return False
            
            # Check minimum data points
            if len(market_data) < self.config.min_data_points:
                tprint_warning(f"⚠️ Insufficient data: {len(market_data)} < {self.config.min_data_points}")
                return False
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = set(required_columns) - set(market_data.columns)
            if missing_columns:
                tprint_warning(f"⚠️ Missing required columns: {missing_columns}")
                return False
            
            # Check for datetime index
            if not isinstance(market_data.index, pd.DatetimeIndex):
                tprint_warning("⚠️ Index must be DatetimeIndex")
                return False
            
            # Check for non-finite values
            if market_data[required_columns].isnull().any().any():
                tprint_warning("⚠️ Data contains null values")
                return False
            
            if not np.isfinite(market_data[required_columns].values).all():
                tprint_warning("⚠️ Data contains non-finite values")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            return False
    
    def _filter_by_quality_thresholds(self, target_result, quality_scores: Dict[str, LabelQualityScore]) -> Any:
        """Filter targets by quality thresholds."""
        try:
            # Create a copy of the target result
            filtered_result = type(target_result)(
                labels=target_result.labels.copy(),
                confidence_scores=target_result.confidence_scores.copy(),
                eligibility_masks=target_result.eligibility_masks.copy()
            )
            
            # Get quality thresholds
            min_auc = self.config.min_auc_threshold
            max_auc_std = self.config.max_auc_std_threshold
            min_psi = self.config.min_psi_threshold
            max_flip_rate = self.config.max_flip_rate_threshold
            min_balance = self.config.min_balance_threshold
            max_balance = self.config.max_balance_threshold
            
            # Filter targets based on quality scores
            valid_targets = []
            for target_name, quality_score in quality_scores.items():
                if (quality_score.auc_mean >= min_auc and
                    quality_score.auc_std <= max_auc_std and
                    quality_score.psi_score <= min_psi and
                    quality_score.flip_rate <= max_flip_rate and
                    min_balance <= quality_score.class_balance <= max_balance):
                    valid_targets.append(target_name)
                else:
                    tprint_warning(f"⚠️ Target {target_name} failed quality thresholds")
            
            # Filter columns based on valid targets
            if valid_targets:
                target_columns = [col for col in filtered_result.labels.columns 
                                if any(target in col for target in valid_targets)]
                filtered_result.labels = filtered_result.labels[target_columns]
                
                conf_columns = [col for col in filtered_result.confidence_scores.columns 
                              if any(target in col for target in valid_targets)]
                filtered_result.confidence_scores = filtered_result.confidence_scores[conf_columns]
                
                mask_columns = [col for col in filtered_result.eligibility_masks.columns 
                              if any(target in col for target in valid_targets)]
                filtered_result.eligibility_masks = filtered_result.eligibility_masks[mask_columns]
            else:
                tprint_warning("⚠️ No targets passed quality thresholds")
                filtered_result.labels = pd.DataFrame()
                filtered_result.confidence_scores = pd.DataFrame()
                filtered_result.eligibility_masks = pd.DataFrame()
            
            return filtered_result
            
        except Exception as e:
            tprint_error(f"❌ Quality filtering failed: {e}")
            return target_result
    
    def _calculate_label_distribution(self, labels: pd.DataFrame) -> Dict[str, Any]:
        """Calculate label distribution statistics."""
        try:
            if labels.empty:
                return {}
            
            distribution = {}
            for col in labels.columns:
                if labels[col].dtype in ['int64', 'float64']:
                    value_counts = labels[col].value_counts()
                    distribution[col] = {
                        'unique_values': len(value_counts),
                        'most_common': value_counts.head(3).to_dict(),
                        'mean': labels[col].mean(),
                        'std': labels[col].std(),
                        'min': labels[col].min(),
                        'max': labels[col].max()
                    }
            
            return distribution
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating label distribution: {e}")
            return {}
    
    def _generate_cache_key(self, market_data: pd.DataFrame) -> str:
        """Generate cache key for market data."""
        try:
            # Simple cache key based on data shape and last timestamp
            data_hash = hash(str(market_data.shape) + str(market_data.index[-1]))
            config_hash = hash(str(self.config))
            return f"volatility_labels_{data_hash}_{config_hash}"
        except Exception:
            return f"volatility_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _generate_data_hash(self, market_data: pd.DataFrame) -> str:
        """Generate hash of market data for intermediate caching."""
        try:
            # Create a hash based on data statistics rather than all data
            stats_hash = hash((
                market_data.shape,
                market_data.index[0].strftime('%Y%m%d'),
                market_data.index[-1].strftime('%Y%m%d'),
                market_data['close'].iloc[0],
                market_data['close'].iloc[-1],
                market_data['volume'].sum()
            ))
            return str(stats_hash)
        except Exception:
            return str(hash(str(market_data.shape)))

    def _cleanup_caches(self):
        """Clean up old cache entries to prevent memory bloat."""
        max_cache_size = 50  # Maximum number of entries per cache

        # Clean main cache
        if len(self.cache) > max_cache_size:
            # Remove oldest entries
            sorted_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            for key in sorted_keys[:-max_cache_size]:
                del self.cache[key]

        # Clean intermediate caches
        for cache in [self.bar_cache, self.volatility_cache, self.noise_cache, self.target_cache, self.quality_cache]:
            if len(cache) > max_cache_size:
                sorted_keys = sorted(cache.keys(), key=lambda k: cache[k].timestamp if hasattr(cache[k], 'timestamp') else datetime.now())
                for key in sorted_keys[:-max_cache_size]:
                    del cache[key]

    def _is_cache_valid(self, cached_result: LabelingResult) -> bool:
        """Check if cached result is still valid."""
        if not self.config.enable_caching:
            return False
        
        cache_age = datetime.now() - cached_result.timestamp
        return cache_age.total_seconds() < (self.config.cache_duration_minutes * 60)
    
    def _create_empty_result(self) -> LabelingResult:
        """Create empty result when processing fails."""
        return LabelingResult(
            labels=pd.DataFrame(),
            confidence_scores=pd.DataFrame(),
            eligibility_masks=pd.DataFrame(),
            quality_scores={},
            config_used=self.config,
            processing_time=0.0
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        if not self.labeling_history:
            return {}
        
        latest_result = self.labeling_history[-1]
        summary = {
            'latest_processing_time': latest_result.processing_time,
            'latest_n_samples': latest_result.n_samples,
            'latest_n_targets': latest_result.n_targets,
            'latest_n_horizons': latest_result.n_horizons,
            'total_runs': len(self.labeling_history),
            'cache_size': len(self.cache)
        }
        
        # Add quality score summary if available
        if latest_result.quality_scores:
            quality_summary = {}
            for target_name, quality_score in latest_result.quality_scores.items():
                quality_summary[target_name] = {
                    'overall_quality': quality_score.overall_quality,
                    'auc_mean': quality_score.auc_mean,
                    'class_balance': quality_score.class_balance
                }
            summary['quality_scores'] = quality_summary
        
        return summary
    
    def save_results(self, result: LabelingResult, output_directory: Optional[str] = None):
        """Save labeling results."""
        try:
            output_dir = output_directory or self.config.output_directory
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Save labels
            labels_path = os.path.join(output_dir, 'volatility_aware_labels.csv')
            result.labels.to_csv(labels_path)
            
            # Save confidence scores
            conf_path = os.path.join(output_dir, 'confidence_scores.csv')
            result.confidence_scores.to_csv(conf_path)
            
            # Save eligibility masks
            mask_path = os.path.join(output_dir, 'eligibility_masks.csv')
            result.eligibility_masks.to_csv(mask_path)
            
            # Save quality scores
            quality_path = os.path.join(output_dir, 'quality_scores.json')
            quality_data = {}
            for target_name, quality_score in result.quality_scores.items():
                quality_data[target_name] = {
                    'predictability': quality_score.predictability,
                    'stability': quality_score.stability,
                    'consistency': quality_score.consistency,
                    'balance': quality_score.balance,
                    'snr_proxy': quality_score.snr_proxy,
                    'overall_quality': quality_score.overall_quality,
                    'auc_mean': quality_score.auc_mean,
                    'auc_std': quality_score.auc_std,
                    'psi_score': quality_score.psi_score,
                    'flip_rate': quality_score.flip_rate,
                    'class_balance': quality_score.class_balance,
                    'mutual_information': quality_score.mutual_information,
                    'information_coefficient': quality_score.information_coefficient,
                    'n_samples': quality_score.n_samples,
                    'processing_time': quality_score.processing_time
                }
            
            import json
            with open(quality_path, 'w') as f:
                json.dump(quality_data, f, indent=2, default=str)
            
            tprint_success(f"💾 Results saved to {output_dir}")
            
        except Exception as e:
            tprint_error(f"❌ Error saving results: {e}")


# Convenience functions
def create_volatility_aware_labeler(config: Optional[VolatilityAwareConfig] = None) -> VolatilityAwareMultiHorizonLabeler:
    """Create volatility-aware labeler with specified configuration."""
    return VolatilityAwareMultiHorizonLabeler(config)


def create_fast_config() -> VolatilityAwareConfig:
    """Create a fast configuration optimized for speed over accuracy."""
    return VolatilityAwareConfig(
        min_data_points=500,
        enable_caching=True,
        cache_duration_minutes=30,
        parallel_processing=True,
        max_workers=2,

        # Simplified bar construction
        bar_construction=BarConstructionConfig(
            bar_type=BarType.DOLLAR,
            bar_size=1000000.0,
            enable_microstructure_filter=False,
            min_bars_required=50
        ),

        # Fast volatility modeling
        volatility=VolatilityConfig(
            method=VolatilityMethod.ATR,
            atr_window=10,
            enable_smoothing=False
        ),

        # Simplified noise gating
        noise_gating=NoiseGatingConfig(
            gate_type=NoiseGateType.MICRO_RANGE,
            enable_micro_range_gating=True,
            enable_variance_ratio_gating=False,
            enable_signal_noise_gating=False
        ),

        # Fast quality scoring
        quality_scoring=QualityScoringConfig(
            baseline_models=['logistic'],
            n_splits=3,
            enable_feature_engineering=False
        ),

        # Simplified multi-target
        multi_target=MultiTargetConfig(
            enable_optimization=False,
            optimization_method='grid',
            n_trials=20,
            max_targets_per_band=1,
            min_lqs_score=0.2
        ),

        # Relaxed thresholds
        min_auc_threshold=0.5,
        max_auc_std_threshold=0.05,
        min_balance_threshold=0.3,
        max_balance_threshold=0.7
    )


def create_accurate_config() -> VolatilityAwareConfig:
    """Create an accurate configuration optimized for quality over speed."""
    return VolatilityAwareConfig(
        min_data_points=2000,
        enable_caching=True,
        cache_duration_minutes=120,
        parallel_processing=True,
        max_workers=None,

        # Comprehensive bar construction
        bar_construction=BarConstructionConfig(
            bar_type=BarType.DOLLAR,
            bar_size=500000.0,
            enable_microstructure_filter=True,
            min_spread_ratio=0.0002,
            min_volume_percentile=5.0,
            max_return_percentile=99.5,
            min_bars_required=200
        ),

        # Comprehensive volatility modeling
        volatility=VolatilityConfig(
            method=VolatilityMethod.COMBINED,
            rv_window=25,
            atr_window=18,
            ewma_alpha=0.08,
            enable_smoothing=True,
            smoothing_window=3
        ),

        # Full noise gating
        noise_gating=NoiseGatingConfig(
            gate_type=NoiseGateType.COMBINED,
            enable_micro_range_gating=True,
            enable_variance_ratio_gating=True,
            enable_signal_noise_gating=True,
            min_snr_ratio=1.5
        ),

        # Comprehensive quality scoring
        quality_scoring=QualityScoringConfig(
            baseline_models=['logistic', 'random_forest'],
            n_splits=5,
            enable_feature_engineering=True,
            feature_window=25,
            n_features=15
        ),

        # Comprehensive multi-target
        multi_target=MultiTargetConfig(
            enable_optimization=True,
            optimization_method='bayesian',
            n_trials=150,
            max_targets_per_band=2,
            min_lqs_score=0.4,
            enable_parallel_processing=True,
            max_workers=4
        ),

        # Strict thresholds
        min_auc_threshold=0.6,
        max_auc_std_threshold=0.02,
        min_balance_threshold=0.4,
        max_balance_threshold=0.6,
        max_correlation_threshold=0.3
    )


def create_balanced_config() -> VolatilityAwareConfig:
    """Create a balanced configuration with good speed/accuracy tradeoff."""
    return VolatilityAwareConfig(
        min_data_points=1000,
        enable_caching=True,
        cache_duration_minutes=60,
        parallel_processing=True,
        max_workers=None,

        # Balanced bar construction
        bar_construction=BarConstructionConfig(
            bar_type=BarType.DOLLAR,
            bar_size=750000.0,
            enable_microstructure_filter=True,
            min_spread_ratio=0.0003,
            min_bars_required=100
        ),

        # Balanced volatility modeling
        volatility=VolatilityConfig(
            method=VolatilityMethod.COMBINED,
            rv_window=20,
            atr_window=14,
            ewma_alpha=0.06,
            enable_smoothing=True
        ),

        # Balanced noise gating
        noise_gating=NoiseGatingConfig(
            gate_type=NoiseGateType.COMBINED,
            enable_micro_range_gating=True,
            enable_variance_ratio_gating=True,
            enable_signal_noise_gating=False
        ),

        # Balanced quality scoring
        quality_scoring=QualityScoringConfig(
            baseline_models=['logistic', 'random_forest'],
            n_splits=4,
            enable_feature_engineering=True,
            feature_window=20,
            n_features=10
        ),

        # Balanced multi-target
        multi_target=MultiTargetConfig(
            enable_optimization=True,
            optimization_method='bayesian',
            n_trials=100,
            max_targets_per_band=2,
            min_lqs_score=0.3,
            enable_parallel_processing=True
        ),

        # Balanced thresholds
        min_auc_threshold=0.55,
        max_auc_std_threshold=0.03,
        min_balance_threshold=0.35,
        max_balance_threshold=0.65,
        max_correlation_threshold=0.4
    )


def generate_volatility_aware_labels(market_data: pd.DataFrame,
                                   config: Optional[VolatilityAwareConfig] = None) -> LabelingResult:
    """Generate volatility-aware labels with default configuration."""
    labeler = VolatilityAwareMultiHorizonLabeler(config)
    return labeler.generate_labels(market_data)


def create_enhanced_analyst_labeler() -> VolatilityAwareMultiHorizonLabeler:
    """Create a labeler optimized for Analyst labels (Should we trade?)."""
    config = VolatilityAwareConfig(
        enable_enhanced_labels=True,
        label_definition_type=LabelDefinitionType.ANALYST,
        analyst_config=AnalystLabelConfig(
            horizon_minutes=60,
            min_profit_threshold_usd=5.0,
            trading_costs=TradingCosts(
                maker_fee=0.001,
                taker_fee=0.002,
                slippage_pct=0.001
            ),
            enable_regime_conditioning=True,
            volatility_scaling_factor=1.0
        ),
        regime_config=RegimeConditionedConfig(
            volatility_scaling_enabled=True,
            base_threshold_multiplier=1.0,
            adaptive_thresholds=True,
            lookback_window=50
        ),
        risk_config=RiskAwareConfig(
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            min_risk_reward_ratio=2.0,
            max_portfolio_risk_pct=0.02
        ),
        cleaning_config=DataCleaningConfig(
            outlier_method="iqr",
            outlier_threshold=3.0,
            min_volume_threshold=1000.0,
            enforce_timestamp_alignment=True
        ),
        stability_config=StabilityCheckConfig(
            recompute_on_refresh=True,
            max_autocorrelation_threshold=0.3,
            enable_oos_balance_check=True,
            balance_tolerance=0.05,
            enable_drift_detection=True,
            drift_threshold=0.1
        )
    )
    return VolatilityAwareMultiHorizonLabeler(config)


def create_enhanced_tactician_labeler() -> VolatilityAwareMultiHorizonLabeler:
    """Create a labeler optimized for Tactician labels (Direction/Magnitude)."""
    config = VolatilityAwareConfig(
        enable_enhanced_labels=True,
        label_definition_type=LabelDefinitionType.TACTICIAN,
        tactician_config=TacticianLabelConfig(
            favorable_excursion_threshold=1.0,
            adverse_excursion_threshold=-2.0,
            horizon_minutes=30,
            enable_regime_conditioning=True,
            volatility_sensitivity=1.0
        ),
        regime_config=RegimeConditionedConfig(
            volatility_scaling_enabled=True,
            base_threshold_multiplier=1.0,
            adaptive_thresholds=True,
            lookback_window=50
        ),
        risk_config=RiskAwareConfig(
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            min_risk_reward_ratio=2.0,
            max_portfolio_risk_pct=0.02
        ),
        cleaning_config=DataCleaningConfig(
            outlier_method="iqr",
            outlier_threshold=3.0,
            min_volume_threshold=1000.0,
            enforce_timestamp_alignment=True
        ),
        stability_config=StabilityCheckConfig(
            recompute_on_refresh=True,
            max_autocorrelation_threshold=0.3,
            enable_oos_balance_check=True,
            balance_tolerance=0.05,
            enable_drift_detection=True,
            drift_threshold=0.1
        )
    )
    return VolatilityAwareMultiHorizonLabeler(config)