"""
Enhanced Data & Labels System - "Define what truth means"

This module implements a comprehensive data and labels system that addresses the core
challenges in trading ML: defining what truth means, cleaning inputs, and ensuring
stability over time.

Key Features:
1. Trading-Aware Label Definitions:
   - Analyst: "Should we trade?" (1 if expected PnL > fees + slippage)
   - Tactician: Direction/magnitude based on max favorable/adverse excursion
   - Regime conditioning: Volatility-scaled thresholds
   - Risk awareness: Label 0 if trade would hit stop before target

2. Comprehensive Data Cleaning:
   - Remove bars with missing/outlier prices/volumes
   - Align timestamps across timeframes
   - De-duplicate overlapping samples from sliding windows
   - Check target shift: verify label distribution doesn't drift

3. Label Stability Monitoring:
   - Recompute labels after every data refresh
   - Track label leakage indicators (autocorrelation)
   - Check OOS label balance similarity to train
   - Apply reweighting when needed

This system is fully integrated with the existing infrastructure and provides
native benefits to all existing components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.metrics import mutual_info_score
import time

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_correlation
)
from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    analyze_nan_values_detailed, validate_dataframe_integrity
)
from src.utils.math_validation import MathValidation

# Import hardware optimization utilities
from src.utils.hardware.m1_gpu_utils import M1GPUManager
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer

# Import data utilities
from src.utils.data.unified_data_utils import UnifiedDataUtils

# Import matrix operations
from src.utils.matrix_operations import UnifiedMatrixOperations

# Import ML common utilities
from src.utils.ml_common.validation.cross_validation import CrossValidator
from src.utils.lookahead_bias_detector import LookaheadBiasDetector, get_global_detector

# Import serialization utilities
from src.utils.serialization_utils import UniversalSerializer

# Import existing components
from .enhanced_label_definitions import (
    EnhancedLabelDefinitions, LabelDefinitionType,
    AnalystLabelConfig, TacticianLabelConfig, RegimeConditionedConfig,
    RiskAwareConfig, DataCleaningConfig, StabilityCheckConfig,
    create_trading_aware_config
)
from .label_balancing import (
    ComprehensiveBalancingSystem, BalancingConfig, WeightingConfig,
    RegimeConfig, ValidationFairnessConfig
)
from src.utils.ml_common.data_processing.data_quality import DataQualityUtilities


class LabelStabilityLevel(Enum):
    """Label stability levels."""
    STABLE = "stable"
    WARNING = "warning"
    CRITICAL = "critical"
    UNSTABLE = "unstable"


class DataQualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class TradingObjectiveConfig:
    """Configuration for trading objectives."""
    
    # Primary objective
    primary_objective: str = "risk_adjusted_returns"  # "returns", "sharpe", "max_drawdown", "risk_adjusted_returns"
    
    # Risk parameters
    max_drawdown_pct: float = 0.05  # 5% max drawdown
    target_sharpe_ratio: float = 1.5
    max_volatility_pct: float = 0.20  # 20% max volatility
    
    # Transaction costs
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.002  # 0.2%
    slippage_pct: float = 0.001  # 0.1%
    
    # Position sizing
    max_position_size_pct: float = 0.10  # 10% max position size
    min_trade_size_usd: float = 100.0
    
    # Regime awareness
    enable_regime_conditioning: bool = True
    regime_volatility_thresholds: Tuple[float, float] = (0.15, 0.25)  # Low, High vol thresholds


@dataclass
class LabelStabilityConfig:
    """Configuration for label stability monitoring."""
    
    # Recomputation settings
    recompute_on_refresh: bool = True
    max_recomputation_gap_hours: int = 24
    force_recomputation_threshold: float = 0.1  # 10% data change
    
    # Leakage detection
    max_autocorrelation_threshold: float = 0.3
    leakage_detection_window: int = 100
    leakage_methods: List[str] = field(default_factory=lambda: ["autocorr", "mutual_info", "granger"])
    
    # OOS balance checking
    enable_oos_balance_check: bool = True
    balance_tolerance: float = 0.05  # 5% tolerance
    min_oos_samples: int = 100
    
    # Drift detection
    enable_drift_detection: bool = True
    drift_threshold: float = 0.1
    drift_detection_method: str = "ks_test"  # "ks_test", "wasserstein", "jensen_shannon"
    
    # Stability thresholds
    stability_warning_threshold: float = 0.7
    stability_critical_threshold: float = 0.5


@dataclass
class EnhancedDataLabelsConfig:
    """Main configuration for enhanced data and labels system."""
    
    # Trading objective
    trading_objective: TradingObjectiveConfig = field(default_factory=TradingObjectiveConfig)
    
    # Label definitions
    label_definitions: Dict[str, Any] = field(default_factory=create_trading_aware_config)
    
    # Data cleaning
    data_cleaning: DataCleaningConfig = field(default_factory=DataCleaningConfig)
    
    # Label stability
    label_stability: LabelStabilityConfig = field(default_factory=LabelStabilityConfig)
    
    # Balancing and weighting
    balancing_config: BalancingConfig = field(default_factory=BalancingConfig)
    weighting_config: WeightingConfig = field(default_factory=WeightingConfig)
    regime_config: RegimeConfig = field(default_factory=RegimeConfig)
    fairness_config: ValidationFairnessConfig = field(default_factory=ValidationFairnessConfig)
    
    # Quality thresholds
    min_data_quality_score: float = 0.7
    min_label_stability_score: float = 0.6
    max_label_imbalance: float = 0.8
    min_capacity_score: float = 0.6
    
    # Performance settings
    enable_caching: bool = True
    cache_duration_hours: int = 6
    parallel_processing: bool = True
    max_workers: Optional[int] = None


class EnhancedDataLabelsSystem:
    """
    Enhanced Data & Labels System - "Define what truth means"
    
    This system implements comprehensive data and labels management that addresses
    the core challenges in trading ML by defining what truth means, cleaning inputs,
    and ensuring stability over time.
    """
    
    def __init__(self, config: Optional[EnhancedDataLabelsConfig] = None):
        """Initialize the enhanced data and labels system."""
        self.config = config or EnhancedDataLabelsConfig()
        self.logger = logging.getLogger('EnhancedDataLabelsSystem')
        
        # Initialize components
        self._initialize_components()

        # Initialize hardware optimization managers
        self.hardware_managers = {
            'gpu': M1GPUManager() if self._is_gpu_available() else None,
            'memory': M1MemoryOptimizer(),
            'cpu': M1CPUOptimizer()
        }

        # Initialize data utilities
        self.data_utils = UnifiedDataUtils()

        # Initialize matrix operations
        self.matrix_ops = UnifiedMatrixOperations()

        # Initialize ML common utilities
        self.cross_validator = CrossValidator()
        self.lookahead_detector = get_global_detector()

        # Initialize serialization utilities
        self.serializer = UniversalSerializer()

        # State tracking
        self.label_history: List[Dict[str, Any]] = []
        self.data_quality_history: List[Dict[str, Any]] = []
        self.stability_history: List[Dict[str, Any]] = []

        # Cache for performance
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        tprint_success("🚀 Enhanced Data & Labels System initialized")
        tprint_info("   → Trading-aware label definitions")
        tprint_info("   → Comprehensive data cleaning")
        tprint_info("   → Label stability monitoring")
        tprint_info("   → Hardware optimization enabled")
        tprint_info("   → Full infrastructure integration")

    def _is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            return self.hardware_managers['gpu'].is_available() if self.hardware_managers['gpu'] else False
        except Exception as e:
            tprint_warning(f"⚠️ Error checking GPU availability: {e}")
            return False
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize enhanced label definitions
            self.label_definitions = EnhancedLabelDefinitions(
                analyst_config=self.config.label_definitions.get('analyst_config'),
                tactician_config=self.config.label_definitions.get('tactician_config'),
                regime_config=self.config.label_definitions.get('regime_config'),
                risk_config=self.config.label_definitions.get('risk_config'),
                cleaning_config=self.config.data_cleaning,
                stability_config=self.config.label_stability
            )
            
            # Initialize data quality utilities
            self.data_quality = DataQualityUtilities({
                'outlier_contamination': 0.1,
                'missing_threshold': 0.5,
                'drift_threshold': self.config.label_stability.drift_threshold,
                'enable_gpu': True,
                'enable_memory_optimization': True
            })
            
            # Initialize balancing system
            self.balancing_system = ComprehensiveBalancingSystem(
                balancing_config=self.config.balancing_config,
                weighting_config=self.config.weighting_config,
                regime_config=self.config.regime_config,
                fairness_config=self.config.fairness_config
            )
            
            tprint_success("✅ All components initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ Component initialization failed: {e}")
            raise
    
    def process_market_data(
        self,
        market_data: pd.DataFrame,
        regime_data: Optional[pd.Series] = None,
        portfolio_state: Optional[Dict[str, Any]] = None,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Process market data through the complete enhanced data and labels pipeline.
        
        Args:
            market_data: OHLCV market data with datetime index
            regime_data: Optional regime assignments
            portfolio_state: Optional current portfolio state
            force_recompute: Force recomputation even if cached
            
        Returns:
            Dictionary containing processed data, labels, and quality metrics
        """
        start_time = time.time()
        tprint_info("🔄 Starting enhanced data and labels processing")
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(market_data, regime_data, portfolio_state)
            if not force_recompute and self._is_cache_valid(cache_key):
                tprint_info("📋 Using cached results")
                # Use deserializer to retrieve cached data
                return self.serializer.deserialize(self.cache[cache_key])
            
            # Step 1: Data Quality Assessment and Cleaning
            tprint_info("🧹 Step 1: Data quality assessment and cleaning")
            data_quality_result = self._assess_and_clean_data(market_data)
            
            if data_quality_result['quality_level'] == DataQualityLevel.CRITICAL:
                tprint_error("❌ Data quality is critical - processing aborted")
                return self._create_error_result("Critical data quality issues")
            
            cleaned_data = data_quality_result['cleaned_data']
            
            # Step 2: Generate Trading-Aware Labels
            tprint_info("🎯 Step 2: Generating trading-aware labels")
            label_result = self._generate_trading_aware_labels(
                cleaned_data, regime_data, portfolio_state
            )
            
            # Step 3: Label Stability Assessment
            tprint_info("🔍 Step 3: Assessing label stability")
            stability_result = self._assess_label_stability(
                label_result['labels'], cleaned_data
            )
            
            # Step 4: Apply Balancing and Weighting
            tprint_info("⚖️ Step 4: Applying balancing and weighting")
            balanced_result = self._apply_balancing_and_weighting(
                cleaned_data, label_result['labels'], label_result['confidence_scores']
            )
            
            # Step 5: Final Quality Check
            tprint_info("✅ Step 5: Final quality check")
            final_quality = self._perform_final_quality_check(
                balanced_result,
                stability_result,
                data_quality_result,
                label_result.get('capacity_diagnostics', {})
            )

            # Compile results
            result = {
                'processed_data': balanced_result['X'],
                'labels': balanced_result['y'],
                'sample_weights': balanced_result['sample_weights'],
                'confidence_scores': label_result['confidence_scores'],
                'capacity_diagnostics': label_result.get('capacity_diagnostics', {}),
                'data_quality': data_quality_result,
                'label_stability': stability_result,
                'final_quality': final_quality,
                'label_metadata': label_result,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now(),
                'cache_key': cache_key
            }
            
            # Store in cache using serialization utilities
            if self.config.enable_caching:
                # Use serializer for efficient caching
                self.cache[cache_key] = self.serializer.serialize(result, compression='auto')
                self.cache_timestamps[cache_key] = datetime.now()
            
            # Update history
            self._update_history(result)
            
            tprint_success(f"✅ Enhanced data and labels processing completed in {result['processing_time']:.2f}s")
            tprint_info(f"   → Data quality: {data_quality_result['quality_level'].value}")
            tprint_info(f"   → Label stability: {stability_result['stability_level'].value}")
            tprint_info(f"   → Final quality: {final_quality['overall_score']:.3f}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Enhanced data and labels processing failed: {e}")
            return self._create_error_result(str(e))
    
    def _assess_and_clean_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality and apply comprehensive cleaning."""
        try:
            tprint_info("🔍 Assessing data quality...")

            # Validate DataFrame integrity using common utilities
            if not validate_dataframe_integrity(market_data):
                tprint_warning("⚠️ DataFrame integrity validation failed")

            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not validate_dataframe_columns(market_data, required_cols):
                tprint_error("❌ Required columns missing from market data")
                return {
                    'original_data': market_data,
                    'cleaned_data': market_data,
                    'quality_level': DataQualityLevel.CRITICAL,
                    'quality_score': 0.0,
                    'error': 'Missing required columns'
                }

            # Comprehensive data quality assessment with hardware optimization
            if self.hardware_managers['memory'].is_memory_efficient(market_data):
                # Use memory-optimized processing
                quality_assessment = self.hardware_managers['memory'].optimized_data_quality_check(
                    market_data, self.data_quality
                )
            else:
                quality_assessment = self.data_quality.calculate_data_quality_score(market_data)
            
            # Enhanced data cleaning with hardware optimization and matrix operations
            tprint_info("🔧 Applying enhanced data cleaning with matrix operations...")

            if self.hardware_managers['cpu'].should_use_optimization(cleaned_data):
                tprint_info("⚡ Using CPU-optimized data cleaning with matrix operations")
                # Use CPU-optimized cleaning
                cleaned_data, cleaning_report = self.hardware_managers['cpu'].optimized_data_cleaning(
                    market_data, self.data_quality, {
                        'missing_value_strategy': 'advanced_imputation',
                        'outlier_method': 'advanced_detection',
                        'correlation_threshold': 0.95,
                        'drift_adaptation': True,
                        'feature_stability_check': True,
                        'use_matrix_operations': True
                    }
                )
            else:
                tprint_info("🔄 Using standard data cleaning with matrix operations")
                cleaned_data, cleaning_report = self.data_quality.enhanced_automated_data_cleaning(
                    market_data, {
                        'missing_value_strategy': 'advanced_imputation',
                        'outlier_method': 'advanced_detection',
                        'correlation_threshold': 0.95,
                        'drift_adaptation': True,
                        'feature_stability_check': True,
                        'use_matrix_operations': True
                    }
                )

            # Additional matrix-based feature correlation analysis
            tprint_info("📊 Performing matrix-based feature correlation analysis...")
            numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                feature_matrix = cleaned_data[numeric_cols].values
                # Use matrix operations for correlation calculation
                correlation_matrix = self.matrix_ops.calculate_pairwise_similarities(
                    feature_matrix.T, method='cosine'
                )
                # Detect highly correlated features using matrix operations
                # Find correlations above threshold (excluding diagonal)
                correlation_flat = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
                high_corr_count = np.sum(correlation_flat > 0.95)
                tprint_info(f"📈 Found {high_corr_count} highly correlated feature pairs")
            else:
                tprint_warning("⚠️ Insufficient numeric columns for correlation analysis")

            # Additional trading-specific cleaning using common utilities
            cleaned_data = safe_dataframe_operation(
                cleaned_data, self._apply_trading_specific_cleaning
            )
            
            # Determine quality level
            quality_score = quality_assessment['overall_score']
            if quality_score >= 0.9:
                quality_level = DataQualityLevel.EXCELLENT
            elif quality_score >= 0.8:
                quality_level = DataQualityLevel.GOOD
            elif quality_score >= 0.7:
                quality_level = DataQualityLevel.FAIR
            elif quality_score >= 0.6:
                quality_level = DataQualityLevel.POOR
            else:
                quality_level = DataQualityLevel.CRITICAL
            
            result = {
                'original_data': market_data,
                'cleaned_data': cleaned_data,
                'quality_assessment': quality_assessment,
                'cleaning_report': cleaning_report,
                'quality_level': quality_level,
                'quality_score': quality_score,
                'samples_removed': len(market_data) - len(cleaned_data),
                'features_removed': len(market_data.columns) - len(cleaned_data.columns)
            }
            
            tprint_success(f"✅ Data cleaning completed: {quality_level.value} quality")
            tprint_info(f"   → Samples: {len(market_data)} → {len(cleaned_data)}")
            tprint_info(f"   → Features: {len(market_data.columns)} → {len(cleaned_data.columns)}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Data quality assessment failed: {e}")
            return {
                'original_data': market_data,
                'cleaned_data': market_data,
                'quality_level': DataQualityLevel.CRITICAL,
                'quality_score': 0.0,
                'error': str(e)
            }
    
    def _apply_trading_specific_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply trading-specific data cleaning rules using common utilities."""
        try:
            cleaned = safe_dataframe_operation(data.copy, lambda x: x.copy())

            # Remove bars with missing OHLCV data using common utilities
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if validate_dataframe_columns(cleaned, required_cols):
                missing_mask = cleaned[required_cols].isnull().any(axis=1)
                cleaned = cleaned[~missing_mask]

                # Remove bars with zero or negative prices using safe operations
                price_cols = ['open', 'high', 'low', 'close']
                if validate_dataframe_columns(cleaned, price_cols):
                    # Use safe operations for price validation
                    invalid_price_mask = safe_dataframe_operation(
                        cleaned[price_cols], lambda x: (x <= 0).any(axis=1)
                    )
                    if not invalid_price_mask.empty:
                        cleaned = cleaned[~invalid_price_mask]

                # Remove bars with zero volume using safe operations
                if 'volume' in cleaned.columns:
                    zero_volume_mask = safe_dataframe_operation(
                        cleaned['volume'], lambda x: x <= 0
                    )
                    if not zero_volume_mask.empty:
                        cleaned = cleaned[~zero_volume_mask]

                # Remove bars with extreme price changes (likely data errors) using matrix operations
                if len(cleaned) > 1:
                    price_changes = safe_dataframe_operation(
                        cleaned['close'].pct_change(), lambda x: x.abs()
                    )
                    if not price_changes.empty:
                        tprint_info("🔍 Detecting extreme price changes using matrix operations...")

                        # Use matrix operations for efficient extreme value detection
                        price_changes_matrix = price_changes.values.reshape(1, -1)
                        # Apply matrix-based outlier detection using statistical methods
                        mean_val = np.mean(price_changes_matrix)
                        std_val = np.std(price_changes_matrix)

                        # Use z-score based detection
                        z_scores = (price_changes_matrix - mean_val) / (std_val + 1e-8)
                        extreme_change_mask = np.abs(z_scores) > 2.0  # 2-sigma rule
                        extreme_change_mask = extreme_change_mask.flatten()

                        extreme_change_mask = pd.Series(extreme_change_mask, index=price_changes.index)
                        extreme_count = extreme_change_mask.sum()

                        if extreme_count > 0:
                            cleaned = cleaned[~extreme_change_mask]
                            tprint_info(f"🚫 Removed {extreme_count} bars with extreme price changes")
                        else:
                            tprint_info("✅ No extreme price changes detected")

                # Ensure proper timestamp alignment using data utilities
                if isinstance(cleaned.index, pd.DatetimeIndex) and len(cleaned) > 0:
                    # Remove duplicate timestamps using common utilities
                    duplicate_mask = safe_dataframe_operation(
                        cleaned.index, lambda x: x.duplicated(keep='first')
                    )
                    if not duplicate_mask.empty:
                        cleaned = cleaned[~duplicate_mask]

                    # Sort by timestamp using data utilities
                    cleaned = safe_dataframe_operation(
                        cleaned, lambda x: x.sort_index()
                    )
            else:
                tprint_error("❌ Required price/volume columns missing for trading-specific cleaning")

            return cleaned

        except Exception as e:
            tprint_warning(f"⚠️ Trading-specific cleaning failed: {e}")
            return data
    
    def _generate_trading_aware_labels(
        self,
        market_data: pd.DataFrame,
        regime_data: Optional[pd.Series] = None,
        portfolio_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate trading-aware labels using enhanced definitions."""
        try:
            tprint_info("🎯 Generating trading-aware labels...")
            
            # Calculate volatility for regime conditioning
            returns = market_data['close'].pct_change().dropna()
            volatility = self._vectorbt_rolling_operation(returns, "std", 20) * np.sqrt(252)  # Annualized
            
            # Generate analyst labels (Should we trade?)
            analyst_labels, analyst_confidence = self.label_definitions.generate_analyst_labels(
                market_data, volatility, regime_data, portfolio_state
            )
            capacity_diagnostics = self.label_definitions.get_latest_analyst_diagnostics()

            # Generate tactician labels (Direction/magnitude)
            tactician_labels, tactician_magnitude = self.label_definitions.generate_tactician_labels(
                market_data, volatility, regime_data, portfolio_state
            )
            
            # Apply regime conditioning if enabled
            if self.config.trading_objective.enable_regime_conditioning and regime_data is not None:
                analyst_labels = self.label_definitions.generate_regime_conditioned_labels(
                    analyst_labels, volatility, regime_data
                )
                tactician_labels = self.label_definitions.generate_regime_conditioned_labels(
                    tactician_labels, volatility, regime_data
                )
            
            # Apply risk awareness
            analyst_labels = self.label_definitions.generate_risk_aware_labels(
                analyst_labels, market_data, portfolio_state
            )
            tactician_labels = self.label_definitions.generate_risk_aware_labels(
                tactician_labels, market_data, portfolio_state
            )

            # Check for lookahead bias using ML common utilities
            lookahead_check = self.lookahead_detector.detect_lookahead_bias(
                market_data, analyst_labels, window_size=10
            )
            if lookahead_check['bias_detected']:
                tprint_warning(f"⚠️ Lookahead bias detected: {lookahead_check['severity']}")

            # Validate label quality using cross-validation utilities
            label_quality = self.cross_validator.validate_labels_cross_validation(
                market_data[['close']], analyst_labels, cv_folds=3
            )
            
            # Create comprehensive labels DataFrame
            labels_df = pd.DataFrame({
                'analyst_label': analyst_labels,
                'analyst_confidence': analyst_confidence,
                'tactician_label': tactician_labels,
                'tactician_magnitude': tactician_magnitude,
                'volatility': volatility,
                'regime': regime_data if regime_data is not None else 'unknown'
            }, index=market_data.index)
            
            # Calculate label statistics
            label_stats = {
                'analyst_positive_ratio': analyst_labels.mean(),
                'tactician_positive_ratio': tactician_labels.mean(),
                'analyst_confidence_mean': analyst_confidence.mean(),
                'tactician_magnitude_mean': tactician_magnitude.mean(),
                'total_samples': len(labels_df),
                'realized_turnover': capacity_diagnostics.get('realized_turnover', 0.0),
                'capacity_score': capacity_diagnostics.get('capacity_score', 1.0),
                'capacity_utilization': capacity_diagnostics.get('capacity_utilization', 0.0),
                'capacity_violations': capacity_diagnostics.get('violations_flagged', False),
                'lookahead_bias_detected': lookahead_check.get('bias_detected', False),
                'lookahead_bias_severity': lookahead_check.get('severity', 'none'),
                'label_quality_score': label_quality.get('overall_score', 0.5)
            }

            result = {
                'labels': labels_df,
                'confidence_scores': pd.DataFrame({
                    'analyst_confidence': analyst_confidence,
                    'tactician_magnitude': tactician_magnitude
                }, index=market_data.index),
                'label_stats': label_stats,
                'volatility_series': volatility,
                'capacity_diagnostics': capacity_diagnostics
            }

            tprint_success(f"✅ Trading-aware labels generated")
            tprint_info(f"   → Analyst positive: {label_stats['analyst_positive_ratio']:.3f}")
            tprint_info(f"   → Tactician positive: {label_stats['tactician_positive_ratio']:.3f}")
            tprint_info(
                "   → Capacity score: "
                f"{label_stats['capacity_score']:.2f} | Turnover: {label_stats['realized_turnover']:.2f}"
            )
            tprint_info(f"   → Total samples: {label_stats['total_samples']}")

            return result
            
        except Exception as e:
            tprint_error(f"❌ Label generation failed: {e}")
            # Return neutral labels on error
            neutral_labels = pd.DataFrame({
                'analyst_label': pd.Series(0, index=market_data.index),
                'analyst_confidence': pd.Series(0.5, index=market_data.index),
                'tactician_label': pd.Series(0, index=market_data.index),
                'tactician_magnitude': pd.Series(1.0, index=market_data.index),
                'volatility': pd.Series(0.2, index=market_data.index),
                'regime': pd.Series('unknown', index=market_data.index)
            })
            return {
                'labels': neutral_labels,
                'confidence_scores': pd.DataFrame({
                    'analyst_confidence': pd.Series(0.5, index=market_data.index),
                    'tactician_magnitude': pd.Series(1.0, index=market_data.index)
                }),
                'label_stats': {'total_samples': len(market_data)},
                'error': str(e)
            }
    
    def _assess_label_stability(
        self,
        labels: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Assess label stability and detect potential issues."""
        try:
            tprint_info("🔍 Assessing label stability...")
            
            # Check for label leakage
            leakage_results = self._detect_label_leakage(labels)
            
            # Check for label drift
            drift_results = self._detect_label_drift(labels)
            
            # Check for autocorrelation
            autocorr_results = self._check_autocorrelation(labels)
            
            # Calculate overall stability score
            stability_components = [
                leakage_results['leakage_score'],
                drift_results['drift_score'],
                autocorr_results['autocorr_score']
            ]
            
            overall_stability = np.mean(stability_components)
            
            # Determine stability level
            if overall_stability >= 0.8:
                stability_level = LabelStabilityLevel.STABLE
            elif overall_stability >= 0.6:
                stability_level = LabelStabilityLevel.WARNING
            elif overall_stability >= 0.4:
                stability_level = LabelStabilityLevel.CRITICAL
            else:
                stability_level = LabelStabilityLevel.UNSTABLE
            
            result = {
                'stability_level': stability_level,
                'overall_stability': overall_stability,
                'leakage_results': leakage_results,
                'drift_results': drift_results,
                'autocorr_results': autocorr_results,
                'stability_components': stability_components,
                'recommendations': self._generate_stability_recommendations(
                    leakage_results, drift_results, autocorr_results
                )
            }
            
            tprint_success(f"✅ Label stability assessment completed: {stability_level.value}")
            tprint_info(f"   → Overall stability: {overall_stability:.3f}")
            tprint_info(f"   → Leakage score: {leakage_results['leakage_score']:.3f}")
            tprint_info(f"   → Drift score: {drift_results['drift_score']:.3f}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Label stability assessment failed: {e}")
            return {
                'stability_level': LabelStabilityLevel.UNSTABLE,
                'overall_stability': 0.0,
                'error': str(e)
            }
    
    def _detect_label_leakage(self, labels: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential label leakage using multiple methods."""
        try:
            leakage_scores = []
            leakage_details = {}
            
            # Method 1: Autocorrelation
            for col in ['analyst_label', 'tactician_label']:
                if col in labels.columns:
                    series = labels[col].dropna()
                    if len(series) > 1:
                        autocorr = series.autocorr(lag=1)
                        if not pd.isna(autocorr):
                            leakage_scores.append(abs(autocorr))
                            leakage_details[f'{col}_autocorr'] = autocorr
            
            # Method 2: Mutual information with future values
            for col in ['analyst_label', 'tactician_label']:
                if col in labels.columns:
                    series = labels[col].dropna()
                    if len(series) > 10:
                        # Check correlation with future values (potential leakage)
                        future_series = series.shift(-1).dropna()
                        if len(future_series) > 5:
                            try:
                                mi_score = mutual_info_score(
                                    series.iloc[:-1], future_series.iloc[:-1]
                                )
                                leakage_scores.append(mi_score)
                                leakage_details[f'{col}_mutual_info'] = mi_score
                            except Exception as e:
                                from src.utils.tprint import tprint_warning
                                tprint_warning(f"⚠️ Failed to calculate mutual information for {col}: {e}")
                                # Use a default high leakage score for failed calculations
                                leakage_scores.append(0.8)
                                leakage_details[f'{col}_mutual_info'] = 0.8
            
            # Calculate overall leakage score (lower is better)
            overall_leakage = 1.0 - np.mean(leakage_scores) if leakage_scores else 1.0
            
            return {
                'leakage_score': overall_leakage,
                'leakage_details': leakage_details,
                'is_leakage_detected': overall_leakage < 0.7
            }
            
        except Exception as e:
            return {
                'leakage_score': 0.5,
                'leakage_details': {},
                'is_leakage_detected': False,
                'error': str(e)
            }
    
    def _detect_label_drift(self, labels: pd.DataFrame) -> Dict[str, Any]:
        """Detect label drift over time."""
        try:
            drift_scores = []
            drift_details = {}
            
            # Split data into early and late periods
            split_point = len(labels) // 2
            early_labels = labels.iloc[:split_point]
            late_labels = labels.iloc[split_point:]
            
            for col in ['analyst_label', 'tactician_label']:
                if col in labels.columns:
                    early_series = early_labels[col].dropna()
                    late_series = late_labels[col].dropna()
                    
                    if len(early_series) > 10 and len(late_series) > 10:
                        # Kolmogorov-Smirnov test
                        try:
                            ks_stat, ks_pvalue = stats.ks_2samp(early_series, late_series)
                            drift_score = 1.0 - ks_pvalue  # Higher score = less drift
                            drift_scores.append(drift_score)
                            drift_details[f'{col}_ks_pvalue'] = ks_pvalue
                        except Exception as e:
                            from src.utils.tprint import tprint_warning

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
                            tprint_warning(f"⚠️ Failed to calculate drift for {col}: {e}")
                            # Use a default moderate drift score for failed calculations
                            drift_scores.append(0.5)
                            drift_details[f'{col}_ks_pvalue'] = 0.5
            
            overall_drift = np.mean(drift_scores) if drift_scores else 1.0
            
            return {
                'drift_score': overall_drift,
                'drift_details': drift_details,
                'is_drift_detected': overall_drift < 0.7
            }
            
        except Exception as e:
            return {
                'drift_score': 0.5,
                'drift_details': {},
                'is_drift_detected': False,
                'error': str(e)
            }
    
    def _check_autocorrelation(self, labels: pd.DataFrame) -> Dict[str, Any]:
        """Check for autocorrelation in labels using matrix operations."""
        try:
            tprint_info("🔍 Checking autocorrelation using matrix operations...")
            autocorr_scores = []
            autocorr_details = {}

            for col in ['analyst_label', 'tactician_label']:
                if col in labels.columns:
                    series = labels[col].dropna()
                    if len(series) > 10:
                        tprint_info(f"📊 Computing autocorrelation for {col}...")

                        # Convert series to matrix for efficient computation
                        values = series.values.reshape(-1, 1)

                        # Check multiple lags using matrix operations
                        lags = [1, 2, 3, 5]
                        lag_scores = []

                        for lag in lags:
                            if len(series) > lag:
                                # Use matrix operations for autocorrelation calculation
                                shifted_values = np.roll(values, lag, axis=0)
                                # Remove the first lag elements where shift occurred
                                valid_values = values[lag:]
                                valid_shifted = shifted_values[lag:]

                                if len(valid_values) > 10:
                                    # Compute correlation using matrix operations
                                    correlation_matrix = self.matrix_ops.calculate_pairwise_similarities(
                                        valid_values, method='cosine'
                                    )
                                    # Extract correlation coefficient (diagonal element)
                                    autocorr = correlation_matrix[0, 0] if correlation_matrix.shape[0] > 0 else 0.0

                                    if not pd.isna(autocorr):
                                        lag_scores.append(abs(autocorr))

                        if lag_scores:
                            avg_autocorr = np.mean(lag_scores)
                            autocorr_scores.append(1.0 - avg_autocorr)  # Lower autocorr = better
                            autocorr_details[f'{col}_avg_autocorr'] = avg_autocorr
                            tprint_info(f"   → {col} autocorrelation: {avg_autocorr:.3f}")

            overall_autocorr = np.mean(autocorr_scores) if autocorr_scores else 1.0

            tprint_success(f"✅ Autocorrelation analysis completed: {overall_autocorr:.3f}")
            return {
                'autocorr_score': overall_autocorr,
                'autocorr_details': autocorr_details,
                'is_high_autocorr': overall_autocorr < 0.7
            }

        except Exception as e:
            tprint_error(f"❌ Autocorrelation check failed: {e}")
            return {
                'autocorr_score': 0.5,
                'autocorr_details': {},
                'is_high_autocorr': False,
                'error': str(e)
            }
    
    def _apply_balancing_and_weighting(
        self,
        market_data: pd.DataFrame,
        labels: pd.DataFrame,
        confidence_scores: pd.DataFrame
    ) -> Dict[str, Any]:
        """Apply comprehensive balancing and weighting."""
        try:
            tprint_info("⚖️ Applying balancing and weighting...")
            
            # Prepare features (use price and volume data) with matrix optimization
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
            X = market_data[feature_cols].copy()

            # Optimize feature matrix for memory and performance
            tprint_info("🔧 Optimizing feature matrix for memory efficiency...")
            if len(X) > 0:
                feature_matrix = X.values
                optimized_matrix = self.matrix_ops.optimize_array(feature_matrix, dtype=np.float32)
                X = pd.DataFrame(optimized_matrix, index=X.index, columns=X.columns)
                tprint_success(f"✅ Feature matrix optimized: {feature_matrix.shape} → {optimized_matrix.shape}")
            
            # Use analyst labels as primary target
            y = labels['analyst_label']
            
            # Prepare additional features for weighting
            additional_features = {
                'volatility': labels.get('volatility', pd.Series(0.2, index=labels.index)),
                'regime': labels.get('regime', pd.Series('unknown', index=labels.index)),
                'confidence': confidence_scores.get('analyst_confidence', pd.Series(0.5, index=labels.index))
            }
            
            # Apply balancing and weighting
            balanced_X, balanced_y, sample_weights = self.balancing_system.balance_and_weight(
                X, y, additional_features=additional_features
            )
            
            result = {
                'X': balanced_X,
                'y': balanced_y,
                'sample_weights': sample_weights,
                'original_size': len(X),
                'balanced_size': len(balanced_X),
                'class_distribution': balanced_y.value_counts().to_dict()
            }
            
            tprint_success(f"✅ Balancing and weighting completed")
            tprint_info(f"   → Original size: {result['original_size']}")
            tprint_info(f"   → Balanced size: {result['balanced_size']}")
            tprint_info(f"   → Class distribution: {result['class_distribution']}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Balancing and weighting failed: {e}")
            # Return original data with uniform weights
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
            X = market_data[feature_cols].copy()
            y = labels['analyst_label']
            sample_weights = pd.Series(1.0, index=X.index)
            
            return {
                'X': X,
                'y': y,
                'sample_weights': sample_weights,
                'error': str(e)
            }
    
    def _perform_final_quality_check(
        self,
        balanced_result: Dict[str, Any],
        stability_result: Dict[str, Any],
        data_quality_result: Dict[str, Any],
        capacity_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform final quality check on all components."""
        try:
            tprint_info("✅ Performing final quality check...")
            
            # Calculate component scores
            data_quality_score = data_quality_result.get('quality_score', 0.0)
            label_stability_score = stability_result.get('overall_stability', 0.0)
            capacity_score = capacity_result.get('capacity_score', 1.0)

            # Check class balance
            class_dist = balanced_result.get('class_distribution', {})
            if class_dist:
                max_class_ratio = max(class_dist.values()) / sum(class_dist.values())
                balance_score = 1.0 - abs(max_class_ratio - 0.5) * 2  # Closer to 0.5 is better
            else:
                balance_score = 0.5
            
            # Calculate overall quality score
            component_scores = [
                data_quality_score,
                label_stability_score,
                balance_score,
                capacity_score
            ]
            overall_score = np.mean(component_scores)
            
            # Determine quality grade
            if overall_score >= 0.9:
                quality_grade = 'A'
            elif overall_score >= 0.8:
                quality_grade = 'B'
            elif overall_score >= 0.7:
                quality_grade = 'C'
            elif overall_score >= 0.6:
                quality_grade = 'D'
            else:
                quality_grade = 'F'
            
            # Generate recommendations with matrix-based analysis
            recommendations = []
            if data_quality_score < 0.7:
                recommendations.append("Improve data quality - address missing values and outliers")
            if label_stability_score < 0.7:
                recommendations.append("Address label stability issues - check for leakage and drift")
            if balance_score < 0.7:
                recommendations.append("Improve class balance - consider different balancing strategies")
            if capacity_score < self.config.min_capacity_score:
                recommendations.append("Capacity violations detected - revisit turnover and holding limits")

            # Add matrix-based dimensionality analysis
            if len(balanced_result['X']) > 0 and len(balanced_result['X'].columns) > 2:
                tprint_info("📊 Performing matrix-based dimensionality analysis...")
                feature_matrix = balanced_result['X'].values

                try:
                    # Use matrix operations for dimensionality analysis
                    # Compute covariance matrix
                    cov_matrix = self.matrix_ops.compute_covariance(feature_matrix)

                    # Perform SVD for dimensionality assessment
                    U, s, Vt = self.matrix_ops.matrix_decomposition(cov_matrix, method='svd')

                    # Analyze explained variance ratio
                    explained_variance_ratio = s / np.sum(s)
                    cumulative_variance = np.cumsum(explained_variance_ratio)

                    # Find number of components needed for 95% variance
                    n_components_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1

                    tprint_info(f"📈 Dimensionality analysis: {n_components_95}/{len(s)} components explain 95% variance")

                    # Add recommendations based on dimensionality analysis
                    if n_components_95 < len(s) * 0.5:
                        recommendations.append(f"Consider dimensionality reduction: {n_components_95} components explain 95% variance")
                    elif n_components_95 > len(s) * 0.8:
                        recommendations.append("High-dimensional data detected - consider feature selection")

                except Exception as e:
                    tprint_warning(f"⚠️ Dimensionality analysis failed: {e}")
                    recommendations.append("Dimensionality analysis unavailable - check data quality")

            result = {
                'overall_score': overall_score,
                'quality_grade': quality_grade,
                'component_scores': {
                    'data_quality': data_quality_score,
                    'label_stability': label_stability_score,
                    'class_balance': balance_score,
                    'capacity': capacity_score
                },
                'recommendations': recommendations,
                'is_acceptable': (
                    overall_score >= self.config.min_data_quality_score and
                    capacity_score >= self.config.min_capacity_score
                )
            }
            
            tprint_success(f"✅ Final quality check completed: {quality_grade} ({overall_score:.3f})")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Final quality check failed: {e}")
            return {
                'overall_score': 0.0,
                'quality_grade': 'F',
                'component_scores': {},
                'recommendations': ['Quality check failed'],
                'is_acceptable': False,
                'error': str(e)
            }
    
    def _generate_stability_recommendations(
        self,
        leakage_results: Dict[str, Any],
        drift_results: Dict[str, Any],
        autocorr_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on stability analysis."""
        recommendations = []
        
        if leakage_results.get('is_leakage_detected', False):
            recommendations.append("Label leakage detected - review feature engineering and labeling logic")
        
        if drift_results.get('is_drift_detected', False):
            recommendations.append("Label drift detected - consider retraining or drift adaptation")
        
        if autocorr_results.get('is_high_autocorr', False):
            recommendations.append("High autocorrelation detected - check for temporal dependencies")
        
        if not recommendations:
            recommendations.append("Labels appear stable - no immediate action required")
        
        return recommendations
    
    def _generate_cache_key(
        self,
        market_data: pd.DataFrame,
        regime_data: Optional[pd.Series] = None,
        portfolio_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for the given inputs."""
        try:
            # Create hash based on data characteristics
            data_hash = hash((
                market_data.shape,
                market_data.index[0] if len(market_data) > 0 else None,
                market_data.index[-1] if len(market_data) > 0 else None,
                market_data['close'].iloc[0] if 'close' in market_data.columns else None,
                market_data['close'].iloc[-1] if 'close' in market_data.columns else None
            ))
            
            regime_hash = hash(str(regime_data)) if regime_data is not None else "none"
            portfolio_hash = hash(str(portfolio_state)) if portfolio_state is not None else "none"
            config_hash = hash(str(self.config))
            
            return f"enhanced_labels_{data_hash}_{regime_hash}_{portfolio_hash}_{config_hash}"
            
        except Exception:
            return f"enhanced_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if not self.config.enable_caching:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_age = datetime.now() - self.cache_timestamps[cache_key]
        return cache_age.total_seconds() < (self.config.cache_duration_hours * 3600)
    
    def _update_history(self, result: Dict[str, Any]):
        """Update processing history."""
        try:
            # Update label history
            self.label_history.append({
                'timestamp': result['timestamp'],
                'label_stats': result.get('label_metadata', {}).get('label_stats', {}),
                'capacity_diagnostics': result.get('capacity_diagnostics', {}),
                'processing_time': result['processing_time']
            })
            
            # Update data quality history
            self.data_quality_history.append({
                'timestamp': result['timestamp'],
                'quality_level': result.get('data_quality', {}).get('quality_level', 'unknown'),
                'quality_score': result.get('data_quality', {}).get('quality_score', 0.0)
            })
            
            # Update stability history
            self.stability_history.append({
                'timestamp': result['timestamp'],
                'stability_level': result.get('label_stability', {}).get('stability_level', 'unknown'),
                'overall_stability': result.get('label_stability', {}).get('overall_stability', 0.0)
            })
            
            # Keep only recent history (last 100 entries)
            for history in [self.label_history, self.data_quality_history, self.stability_history]:
                if len(history) > 100:
                    history[:] = history[-100:]
                    
        except Exception as e:
            tprint_warning(f"⚠️ Failed to update history: {e}")
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure."""
        return {
            'error': error_message,
            'processed_data': pd.DataFrame(),
            'labels': pd.DataFrame(),
            'sample_weights': pd.Series(),
            'confidence_scores': pd.DataFrame(),
            'data_quality': {'quality_level': DataQualityLevel.CRITICAL},
            'label_stability': {'stability_level': LabelStabilityLevel.UNSTABLE},
            'final_quality': {'overall_score': 0.0, 'is_acceptable': False},
            'processing_time': 0.0,
            'timestamp': datetime.now()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and performance metrics."""
        try:
            status = {
                'system_initialized': True,
                'cache_size': len(self.cache),
                'history_size': {
                    'labels': len(self.label_history),
                    'data_quality': len(self.data_quality_history),
                    'stability': len(self.stability_history)
                },
                'config': {
                    'trading_objective': self.config.trading_objective.primary_objective,
                    'enable_caching': self.config.enable_caching,
                    'parallel_processing': self.config.parallel_processing
                }
            }
            
            # Add recent performance metrics
            if self.label_history:
                recent_labels = self.label_history[-1]
                status['recent_processing_time'] = recent_labels.get('processing_time', 0.0)
            
            if self.data_quality_history:
                recent_quality = self.data_quality_history[-1]
                status['recent_quality_score'] = recent_quality.get('quality_score', 0.0)
            
            if self.stability_history:
                recent_stability = self.stability_history[-1]
                status['recent_stability_score'] = recent_stability.get('overall_stability', 0.0)
            
            return status
            
        except Exception as e:
            return {
                'system_initialized': False,
                'error': str(e)
            }
    
    def clear_cache(self):
        """Clear all cached results."""
        self.cache.clear()
        self.cache_timestamps.clear()
        tprint_info("🗑️ Cache cleared")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all processing runs."""
        try:
            if not self.label_history:
                return {'message': 'No processing history available'}
            
            # Calculate performance metrics
            processing_times = [entry['processing_time'] for entry in self.label_history]
            quality_scores = [entry['quality_score'] for entry in self.data_quality_history]
            stability_scores = [entry['overall_stability'] for entry in self.stability_history]
            
            summary = {
                'total_runs': len(self.label_history),
                'avg_processing_time': np.mean(processing_times),
                'avg_quality_score': np.mean(quality_scores) if quality_scores else 0.0,
                'avg_stability_score': np.mean(stability_scores) if stability_scores else 0.0,
                'cache_hit_rate': len(self.cache) / max(1, len(self.label_history)),
                'recent_trends': {
                    'processing_time_trend': 'stable',  # Could implement trend analysis
                    'quality_trend': 'stable',
                    'stability_trend': 'stable'
                }
            }
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}


# Convenience functions for easy usage
def create_enhanced_data_labels_system(
    config: Optional[EnhancedDataLabelsConfig] = None
) -> EnhancedDataLabelsSystem:
    """Create enhanced data and labels system with specified configuration."""
    return EnhancedDataLabelsSystem(config)


def create_trading_optimized_config() -> EnhancedDataLabelsConfig:
    """Create configuration optimized for trading objectives."""
    return EnhancedDataLabelsConfig(
        trading_objective=TradingObjectiveConfig(
            primary_objective="risk_adjusted_returns",
            max_drawdown_pct=0.05,
            target_sharpe_ratio=1.5,
            enable_regime_conditioning=True
        ),
        label_stability=LabelStabilityConfig(
            recompute_on_refresh=True,
            max_autocorrelation_threshold=0.2,
            enable_drift_detection=True,
            drift_threshold=0.05
        ),
        min_data_quality_score=0.8,
        min_label_stability_score=0.7
    )


def create_research_optimized_config() -> EnhancedDataLabelsConfig:
    """Create configuration optimized for research and experimentation."""
    return EnhancedDataLabelsConfig(
        trading_objective=TradingObjectiveConfig(
            primary_objective="returns",
            enable_regime_conditioning=True
        ),
        label_stability=LabelStabilityConfig(
            recompute_on_refresh=False,
            max_autocorrelation_threshold=0.3,
            enable_drift_detection=True,
            drift_threshold=0.1
        ),
        min_data_quality_score=0.6,
        min_label_stability_score=0.5
    )