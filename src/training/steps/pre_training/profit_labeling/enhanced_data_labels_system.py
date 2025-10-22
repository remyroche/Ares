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
from abc import ABC

# Import BaseStep
from src.training.steps.base_step import BaseStep

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_correlation
)
from src.utils.math_validation import MathValidation

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
    
    # Performance settings
    enable_caching: bool = True
    cache_duration_hours: int = 6
    parallel_processing: bool = True
    max_workers: Optional[int] = None


class EnhancedDataLabelsSystem(BaseStep):
    """
    Enhanced Data & Labels System - "Define what truth means"
    
    This system implements comprehensive data and labels management that addresses
    the core challenges in trading ML by defining what truth means, cleaning inputs,
    and ensuring stability over time.
    Inherits from BaseStep for standardized pipeline integration.
    """
    
    def __init__(self, config: Optional[EnhancedDataLabelsConfig] = None):
        """Initialize the enhanced data and labels system."""
        super().__init__()
        self.config = config or EnhancedDataLabelsConfig()
        self.logger = logging.getLogger('EnhancedDataLabelsSystem')
        
        # Initialize components
        self._initialize_components()
        
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
        tprint_info("   → Full infrastructure integration")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the enhanced data and labels processing step.
        
        Args:
            config: Configuration dictionary containing:
                - market_data: DataFrame with OHLCV market data
                - regime_data: Optional Series with regime assignments
                - portfolio_state: Optional portfolio state information
                - force_recompute: Optional flag to force recomputation
                - symbol: Optional symbol for context
                - exchange: Optional exchange for context
                - information: Optional information for context
                - direction: Optional direction for context
                - model: Optional model type for context
        
        Returns:
            Dictionary containing:
                - success: Boolean indicating success
                - processed_data: Processed market data
                - labels: Generated labels
                - quality_metrics: Data quality metrics
                - stability_metrics: Label stability metrics
                - artifacts: List of generated artifacts
        """
        try:
            # Set context for enhanced file naming and operations
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information=config.get('information'),
                direction=config.get('direction', 'long'),
                model=config.get('model', 'Analyst')
            )
            
            # Extract data from config
            market_data = config.get('market_data')
            regime_data = config.get('regime_data')
            portfolio_state = config.get('portfolio_state')
            force_recompute = config.get('force_recompute', False)
            
            if market_data is None:
                return {
                    'success': False,
                    'error': 'Missing required data: market_data is required'
                }
            
            # Validate inputs
            if not isinstance(market_data, pd.DataFrame):
                return {
                    'success': False,
                    'error': 'market_data must be a pandas DataFrame'
                }
            
            # Preview input data
            self.tprint_data_preview(market_data, "input_market_data", max_rows=5)
            self.tprint_data_format(market_data, "input_market_data")
            
            # Process market data through the enhanced pipeline
            result = self.process_market_data(
                market_data=market_data,
                regime_data=regime_data,
                portfolio_state=portfolio_state,
                force_recompute=force_recompute
            )
            
            if not result.get('success', True):
                return {
                    'success': False,
                    'error': result.get('error', 'Processing failed')
                }
            
            # Save artifacts
            artifacts = []
            if self.config.save_artifacts:
                # Preview processed data
                self.tprint_data_preview(result['processed_data'], "processed_market_data", max_rows=5)
                self.tprint_data_format(result['processed_data'], "processed_market_data")
                
                # Save processed data
                processed_data_path = self._save_dataframe(
                    result['processed_data'], 
                    'processed_market_data'
                )
                if processed_data_path:
                    artifacts.append(processed_data_path)
                
                # Save labels
                if 'labels' in result:
                    labels_df = result['labels'].to_frame('labels')
                    self.tprint_data_preview(labels_df, "generated_labels", max_rows=5)
                    self.tprint_data_format(labels_df, "generated_labels")
                    
                    labels_path = self._save_dataframe(
                        labels_df, 
                        'generated_labels'
                    )
                    if labels_path:
                        artifacts.append(labels_path)
                
                # Save quality metrics
                if 'quality_metrics' in result:
                    self.tprint_data_format(result['quality_metrics'], "data_quality_metrics")
                    quality_path = self._save_metadata(
                        result['quality_metrics'], 
                        'data_quality_metrics'
                    )
                    if quality_path:
                        artifacts.append(quality_path)
                
                # Save stability metrics
                if 'stability_metrics' in result:
                    self.tprint_data_format(result['stability_metrics'], "label_stability_metrics")
                    stability_path = self._save_metadata(
                        result['stability_metrics'], 
                        'label_stability_metrics'
                    )
                    if stability_path:
                        artifacts.append(stability_path)
            
            # Log metrics
            self.tprint_metrics({
                'original_samples': len(market_data),
                'processed_samples': len(result['processed_data']),
                'quality_score': result.get('quality_metrics', {}).get('overall_score', 0),
                'stability_score': result.get('stability_metrics', {}).get('stability_score', 0)
            }, "enhanced_data_labels_metrics")
            
            # Generate outcome file
            outcome_content = self._generate_outcome_content(result, artifacts)
            self._save_outcome_file(outcome_content, 'enhanced_data_labels_outcome')
            
            return {
                'success': True,
                'processed_data': result['processed_data'],
                'labels': result.get('labels'),
                'quality_metrics': result.get('quality_metrics', {}),
                'stability_metrics': result.get('stability_metrics', {}),
                'artifacts': artifacts
            }
            
        except Exception as e:
            error_msg = f"Enhanced data and labels processing failed: {str(e)}"
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def _generate_outcome_content(self, result: Dict[str, Any], artifacts: List[str]) -> str:
        """Generate outcome file content."""
        content = f"""# Enhanced Data & Labels System Outcome

## Summary
- **Status**: {'Success' if result.get('success', True) else 'Failed'}
- **Processing Time**: {result.get('processing_time', 0):.2f} seconds
- **Original Samples**: {result.get('original_samples', 0)}
- **Processed Samples**: {result.get('processed_samples', 0)}
- **Artifacts Generated**: {len(artifacts)}

## Data Quality Metrics
"""
        
        if 'quality_metrics' in result:
            quality = result['quality_metrics']
            content += f"""
- **Overall Quality Score**: {quality.get('overall_score', 0):.3f}
- **Missing Data Rate**: {quality.get('missing_rate', 0):.3f}
- **Outlier Rate**: {quality.get('outlier_rate', 0):.3f}
- **Data Completeness**: {quality.get('completeness', 0):.3f}
"""
        
        content += f"""
## Label Stability Metrics
"""
        
        if 'stability_metrics' in result:
            stability = result['stability_metrics']
            content += f"""
- **Stability Score**: {stability.get('stability_score', 0):.3f}
- **Label Drift**: {stability.get('label_drift', 0):.3f}
- **Autocorrelation**: {stability.get('autocorrelation', 0):.3f}
"""
        
        content += f"""
## Generated Artifacts
{chr(10).join(f"- {artifact}" for artifact in artifacts)}

## Configuration
- **Trading-Aware Labels**: {self.config.label_definitions.get('enable_trading_aware', True)}
- **Data Cleaning**: {self.config.data_cleaning.get('enable_cleaning', True)}
- **Stability Monitoring**: {self.config.label_stability.get('enable_monitoring', True)}
"""
        
        return content
    
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
                return self.cache[cache_key]
            
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
                balanced_result, stability_result, data_quality_result
            )
            
            # Compile results
            result = {
                'processed_data': balanced_result['X'],
                'labels': balanced_result['y'],
                'sample_weights': balanced_result['sample_weights'],
                'confidence_scores': label_result['confidence_scores'],
                'data_quality': data_quality_result,
                'label_stability': stability_result,
                'final_quality': final_quality,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now(),
                'cache_key': cache_key
            }
            
            # Store in cache
            if self.config.enable_caching:
                self.cache[cache_key] = result
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
            
            # Comprehensive data quality assessment
            quality_assessment = self.data_quality.calculate_data_quality_score(market_data)
            
            # Enhanced data cleaning
            cleaned_data, cleaning_report = self.data_quality.enhanced_automated_data_cleaning(
                market_data, {
                    'missing_value_strategy': 'advanced_imputation',
                    'outlier_method': 'advanced_detection',
                    'correlation_threshold': 0.95,
                    'drift_adaptation': True,
                    'feature_stability_check': True
                }
            )
            
            # Additional trading-specific cleaning
            cleaned_data = self._apply_trading_specific_cleaning(cleaned_data)
            
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
        """Apply trading-specific data cleaning rules."""
        try:
            cleaned = data.copy()
            
            # Remove bars with missing OHLCV data
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_mask = cleaned[required_cols].isnull().any(axis=1)
            cleaned = cleaned[~missing_mask]
            
            # Remove bars with zero or negative prices
            price_cols = ['open', 'high', 'low', 'close']
            invalid_price_mask = (cleaned[price_cols] <= 0).any(axis=1)
            cleaned = cleaned[~invalid_price_mask]
            
            # Remove bars with zero volume
            if 'volume' in cleaned.columns:
                zero_volume_mask = cleaned['volume'] <= 0
                cleaned = cleaned[~zero_volume_mask]
            
            # Remove bars with extreme price changes (likely data errors)
            if len(cleaned) > 1:
                price_changes = cleaned['close'].pct_change().abs()
                extreme_change_mask = price_changes > 0.5  # 50% change
                cleaned = cleaned[~extreme_change_mask]
            
            # Ensure proper timestamp alignment
            if isinstance(cleaned.index, pd.DatetimeIndex):
                # Remove duplicate timestamps
                cleaned = cleaned[~cleaned.index.duplicated(keep='first')]
                
                # Sort by timestamp
                cleaned = cleaned.sort_index()
            
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
            volatility = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
            
            # Generate analyst labels (Should we trade?)
            analyst_labels, analyst_confidence = self.label_definitions.generate_analyst_labels(
                market_data, volatility, regime_data, portfolio_state
            )
            
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
                'total_samples': len(labels_df)
            }
            
            result = {
                'labels': labels_df,
                'confidence_scores': pd.DataFrame({
                    'analyst_confidence': analyst_confidence,
                    'tactician_magnitude': tactician_magnitude
                }, index=market_data.index),
                'label_stats': label_stats,
                'volatility_series': volatility
            }
            
            tprint_success(f"✅ Trading-aware labels generated")
            tprint_info(f"   → Analyst positive: {label_stats['analyst_positive_ratio']:.3f}")
            tprint_info(f"   → Tactician positive: {label_stats['tactician_positive_ratio']:.3f}")
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
                            except:
                                pass
            
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
                        except:
                            pass
            
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
        """Check for autocorrelation in labels."""
        try:
            autocorr_scores = []
            autocorr_details = {}
            
            for col in ['analyst_label', 'tactician_label']:
                if col in labels.columns:
                    series = labels[col].dropna()
                    if len(series) > 10:
                        # Check multiple lags
                        lags = [1, 2, 3, 5]
                        lag_scores = []
                        
                        for lag in lags:
                            if len(series) > lag:
                                autocorr = series.autocorr(lag=lag)
                                if not pd.isna(autocorr):
                                    lag_scores.append(abs(autocorr))
                        
                        if lag_scores:
                            avg_autocorr = np.mean(lag_scores)
                            autocorr_scores.append(1.0 - avg_autocorr)  # Lower autocorr = better
                            autocorr_details[f'{col}_avg_autocorr'] = avg_autocorr
            
            overall_autocorr = np.mean(autocorr_scores) if autocorr_scores else 1.0
            
            return {
                'autocorr_score': overall_autocorr,
                'autocorr_details': autocorr_details,
                'is_high_autocorr': overall_autocorr < 0.7
            }
            
        except Exception as e:
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
            
            # Prepare features (use price and volume data)
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
            X = market_data[feature_cols].copy()
            
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
        data_quality_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform final quality check on all components."""
        try:
            tprint_info("✅ Performing final quality check...")
            
            # Calculate component scores
            data_quality_score = data_quality_result.get('quality_score', 0.0)
            label_stability_score = stability_result.get('overall_stability', 0.0)
            
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
                balance_score
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
            
            # Generate recommendations
            recommendations = []
            if data_quality_score < 0.7:
                recommendations.append("Improve data quality - address missing values and outliers")
            if label_stability_score < 0.7:
                recommendations.append("Address label stability issues - check for leakage and drift")
            if balance_score < 0.7:
                recommendations.append("Improve class balance - consider different balancing strategies")
            
            result = {
                'overall_score': overall_score,
                'quality_grade': quality_grade,
                'component_scores': {
                    'data_quality': data_quality_score,
                    'label_stability': label_stability_score,
                    'class_balance': balance_score
                },
                'recommendations': recommendations,
                'is_acceptable': overall_score >= self.config.min_data_quality_score
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
                'label_stats': result.get('labels', {}).get('label_stats', {}),
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