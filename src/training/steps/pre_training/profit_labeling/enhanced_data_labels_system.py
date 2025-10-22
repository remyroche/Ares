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
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Literal
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

# Note: tprint and hardware utilities are available through BaseStep
# No need for direct imports as they're inherited from BaseStep

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
    
    # Enable monitoring flag
    enable_monitoring: bool = True


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
    
    # Artifact management
    save_artifacts: bool = True


class EnhancedDataLabelsSystem(BaseStep):
    """
    Enhanced Data & Labels System - "Define what truth means"
    
    This system implements comprehensive data and labels management that addresses
    the core challenges in trading ML by defining what truth means, cleaning inputs,
    and ensuring stability over time.
    Inherits from BaseStep for standardized pipeline integration.
    """
    
    def __init__(self, config: Optional[EnhancedDataLabelsConfig] = None, random_seed: Optional[int] = None):
        """Initialize the enhanced data and labels system."""
        super().__init__()
        self.config = config or EnhancedDataLabelsConfig()
        self.logger = logging.getLogger('EnhancedDataLabelsSystem')
        
        # Set random seed for reproducibility
        self.random_seed = random_seed or 42
        self._set_random_seed()
        
        # Initialize components
        self._initialize_components()
        
        # State tracking
        self.label_history: List[Dict[str, Any]] = []
        self.data_quality_history: List[Dict[str, Any]] = []
        self.stability_history: List[Dict[str, Any]] = []
        
        # Cache for performance
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Reproducibility tracking
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.random_seed}"
        
        self.tprint_success("🚀 Enhanced Data & Labels System initialized")
        self.tprint_info("   → Trading-aware label definitions")
        self.tprint_info("   → Comprehensive data cleaning")
        self.tprint_info("   → Label stability monitoring")
        self.tprint_info("   → Full infrastructure integration")
        self.tprint_info(f"   → Random seed: {self.random_seed}")
        self.tprint_info(f"   → Run ID: {self.run_id}")
    
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
            is_valid, validation_msg = self._validate_input_data(market_data)
            if not is_valid:
                return {
                    'success': False,
                    'error': f'Input validation failed: {validation_msg}'
                }
            
            # Preview input data
            self.tprint_data_preview(market_data, "input_market_data", max_rows=5)
            self.tprint_data_format(market_data, "input_market_data")
            
            # Apply hardware optimization to input data
            if self.hardware_utils and self.hardware_utils.get('optimize_dataframe'):
                market_data = self.hardware_utils['optimize_dataframe'](market_data)
                self.tprint_info("🔧 Input data optimized for hardware acceleration")
            
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
                # Apply hardware optimization to processed data
                if self.hardware_utils and self.hardware_utils.get('optimize_dataframe'):
                    result['processed_data'] = self.hardware_utils['optimize_dataframe'](result['processed_data'])
                    self.tprint_info("🔧 Processed data optimized for hardware acceleration")
            
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
            
            # Check monitoring alerts
            monitoring_alerts = self.check_monitoring_alerts(result)
            result['monitoring_alerts'] = monitoring_alerts
            
            # Log metrics
            self.tprint_metrics({
                'original_samples': len(market_data),
                'processed_samples': len(result['processed_data']),
                'quality_score': result.get('quality_metrics', {}).get('overall_score', 0),
                'stability_score': result.get('stability_metrics', {}).get('stability_score', 0),
                'alert_count': monitoring_alerts.get('alert_count', 0),
                'overall_status': monitoring_alerts.get('overall_status', 'unknown')
            }, "enhanced_data_labels_metrics")
            
            # Generate outcome file
            outcome_content = self._generate_outcome_content(result, artifacts)
            self._save_outcome_file(outcome_content, 'enhanced_data_labels_outcome')
            
            return {
                'success': True,
                'processed_data': result['processed_data'],
                'labels': result.get('labels'),
                'sample_weights': result.get('sample_weights'),
                'quality_metrics': result.get('quality_metrics', {}),
                'stability_metrics': result.get('stability_metrics', {}),
                'label_stats': result.get('label_stats', {}),
                'processing_time': result.get('processing_time', 0.0),
                'original_samples': result.get('original_samples', 0),
                'processed_samples': result.get('processed_samples', 0),
                'artifacts': artifacts,
                'run_id': self.run_id,
                'manifest': self.generate_label_manifest(result)
            }
            
        except Exception as e:
            error_msg = f"Enhanced data and labels processing failed: {str(e)}"
            self.tprint_error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def _generate_outcome_content(self, result: Dict[str, Any], artifacts: List[str]) -> str:
        """Generate comprehensive outcome file content with detailed reporting."""
        content = f"""# Enhanced Data & Labels System Outcome Report

## Executive Summary
- **Status**: {'✅ Success' if result.get('success', True) else '❌ Failed'}
- **Processing Time**: {result.get('processing_time', 0):.2f} seconds
- **Run ID**: {result.get('run_id', 'unknown')}
- **Timestamp**: {result.get('timestamp', datetime.now()).isoformat() if isinstance(result.get('timestamp'), datetime) else str(result.get('timestamp', 'unknown'))}
- **Original Samples**: {result.get('original_samples', 0):,}
- **Processed Samples**: {result.get('processed_samples', 0):,}
- **Data Reduction**: {((result.get('original_samples', 0) - result.get('processed_samples', 0)) / max(result.get('original_samples', 1), 1) * 100):.1f}%
- **Artifacts Generated**: {len(artifacts)}

## Data Quality Assessment
"""
        
        if 'quality_metrics' in result:
            quality = result['quality_metrics']
            content += f"""
### Overall Quality Metrics
- **Overall Quality Score**: {quality.get('overall_score', 0):.3f} / 1.000
- **Quality Grade**: {quality.get('quality_grade', 'F')}
- **Quality Level**: {quality.get('quality_level', 'unknown')}
- **Missing Data Rate**: {quality.get('missing_rate', 0):.3f} ({quality.get('missing_rate', 0)*100:.1f}%)
- **Outlier Rate**: {quality.get('outlier_rate', 0):.3f} ({quality.get('outlier_rate', 0)*100:.1f}%)
- **Data Completeness**: {quality.get('completeness', 0):.3f} ({quality.get('completeness', 0)*100:.1f}%)
- **Effective Sample Size**: {quality.get('effective_sample_size', 0):,.0f}

### Data Cleaning Results
- **Samples Removed**: {quality.get('samples_removed', 0):,}
- **Features Removed**: {quality.get('features_removed', 0)}
- **Session Gap Anomalies**: {quality.get('session_gap_anomalies', 0)}
- **Duplicate Count**: {quality.get('dedup_count', 0)}
"""
        
        content += f"""
## Label Stability Analysis
"""
        
        if 'stability_metrics' in result:
            stability = result['stability_metrics']
            content += f"""
### Stability Metrics
- **Overall Stability Score**: {stability.get('stability_score', 0):.3f} / 1.000
- **Stability Level**: {stability.get('stability_level', 'unknown')}
- **Label Drift Score**: {stability.get('label_drift', 0):.3f}
- **Autocorrelation Score**: {stability.get('autocorrelation', 0):.3f}
- **Leakage Detected**: {'⚠️ Yes' if stability.get('leakage_detected', False) else '✅ No'}

### Stability Recommendations
{chr(10).join(f"- {rec}" for rec in stability.get('recommendations', ['No recommendations available']))}
"""
        
        content += f"""
## Label Statistics
"""
        
        if 'label_stats' in result:
            label_stats = result['label_stats']
            content += f"""
### Label Distribution
- **Total Samples**: {label_stats.get('total_samples', 0):,}
- **Analyst Positive Ratio**: {label_stats.get('analyst_positive_ratio', 0):.3f} ({label_stats.get('analyst_positive_ratio', 0)*100:.1f}%)
- **Tactician Positive Ratio**: {label_stats.get('tactician_positive_ratio', 0):.3f} ({label_stats.get('tactician_positive_ratio', 0)*100:.1f}%)
- **Analyst Confidence Mean**: {label_stats.get('analyst_confidence_mean', 0):.3f}
- **Tactician Magnitude Mean**: {label_stats.get('tactician_magnitude_mean', 0):.3f}
"""
        
        content += f"""
## Monitoring & Alerts
"""
        
        if 'monitoring_alerts' in result:
            alerts = result['monitoring_alerts']
            content += f"""
### Alert Status
- **Overall Status**: {alerts.get('overall_status', 'unknown').upper()}
- **Total Alerts**: {alerts.get('alert_count', 0)}

### Alert Details
"""
            
            for alert_type, alert_list in alerts.items():
                if isinstance(alert_list, list) and alert_list:
                    content += f"\n#### {alert_type.replace('_', ' ').title()}\n"
                    for alert in alert_list:
                        content += f"- {alert}\n"
        
        content += f"""
## System Configuration
- **Trading-Aware Labels**: {self.config.label_definitions.get('enable_trading_aware', True)}
- **Data Cleaning**: {self.config.data_cleaning.enable_cleaning}
- **Stability Monitoring**: {self.config.label_stability.enable_monitoring}
- **Caching Enabled**: {self.config.enable_caching}
- **Random Seed**: {getattr(self, 'random_seed', 'unknown')}
- **Min Data Quality Score**: {self.config.min_data_quality_score}
- **Min Label Stability Score**: {self.config.min_label_stability_score}

## Generated Artifacts
{chr(10).join(f"- {artifact}" for artifact in artifacts)}

## Label Manifest
"""
        
        if 'manifest' in result:
            manifest = result['manifest']
            content += f"""
- **Data Checksum**: {manifest.get('data_checksum', 'unknown')}
- **Label Checksum**: {manifest.get('label_checksum', 'unknown')}
- **Config Hash**: {manifest.get('config_hash', 'unknown')}
- **Version**: {manifest.get('version', 'unknown')}
- **Quality Score**: {manifest.get('quality_score', 0):.3f}
- **Stability Score**: {manifest.get('stability_score', 0):.3f}
"""
        
        content += f"""
## Recommendations
"""
        
        # Generate recommendations based on all metrics
        recommendations = self._generate_comprehensive_recommendations(result)
        for i, rec in enumerate(recommendations, 1):
            content += f"{i}. {rec}\n"
        
        content += f"""
---
*Report generated by Enhanced Data & Labels System v1.0.0*
*Generated at: {datetime.now().isoformat()}*
"""
        
        return content
    
    def _generate_comprehensive_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations based on all metrics."""
        recommendations = []
        
        # Quality-based recommendations
        quality_score = result.get('quality_metrics', {}).get('overall_score', 0)
        if quality_score < 0.7:
            recommendations.append("Improve data quality by addressing missing values and outliers")
        elif quality_score < 0.8:
            recommendations.append("Consider additional data cleaning to improve quality score")
        
        # Stability-based recommendations
        stability_score = result.get('stability_metrics', {}).get('stability_score', 0)
        if stability_score < 0.7:
            recommendations.append("Address label stability issues - check for leakage and drift")
        
        # Sample size recommendations
        processed_samples = result.get('processed_samples', 0)
        if processed_samples < 1000:
            recommendations.append("Consider collecting more data to improve model performance")
        
        # Alert-based recommendations
        alerts = result.get('monitoring_alerts', {})
        if alerts.get('alert_count', 0) > 0:
            recommendations.append("Review and address monitoring alerts to improve system health")
        
        # Balance recommendations
        label_stats = result.get('label_stats', {})
        analyst_ratio = label_stats.get('analyst_positive_ratio', 0.5)
        if analyst_ratio < 0.1 or analyst_ratio > 0.9:
            recommendations.append("Consider adjusting label thresholds to improve class balance")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System is performing well - continue monitoring for any changes")
        
        return recommendations
    
    def _set_random_seed(self):
        """Set random seed for reproducibility."""
        try:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
            
            # Set seed for any other libraries that support it
            try:
                import torch
                torch.manual_seed(self.random_seed)
            except ImportError:
                pass
            
            try:
                import tensorflow as tf
                tf.random.set_seed(self.random_seed)
            except ImportError:
                pass
            
            self.tprint_info(f"🔧 Random seed set to {self.random_seed}")
            
        except Exception as e:
            self.tprint_warning(f"⚠️ Failed to set random seed: {e}")
    
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
            
            # Initialize balancing system with random seed
            self.config.balancing_config.random_state = self.random_seed
            self.balancing_system = ComprehensiveBalancingSystem(
                balancing_config=self.config.balancing_config,
                weighting_config=self.config.weighting_config,
                regime_config=self.config.regime_config,
                fairness_config=self.config.fairness_config
            )
            
            self.tprint_success("✅ All components initialized successfully")
            
        except Exception as e:
            self.tprint_error(f"❌ Component initialization failed: {e}")
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
        self.tprint_info("🔄 Starting enhanced data and labels processing")
        
        try:
            # Validate inputs
            is_valid, validation_msg = self._validate_input_data(market_data)
            if not is_valid:
                return self._create_error_result(f"Input validation failed: {validation_msg}")
            
            # Check cache first
            cache_key = self._generate_cache_key(market_data, regime_data, portfolio_state)
            if not force_recompute and self._is_cache_valid(cache_key):
                self.tprint_info("📋 Using cached results")
                cached_result = self.cache[cache_key]
                # Ensure cached result has all required fields
                return self._ensure_result_completeness(cached_result)
            
            # Step 1: Data Quality Assessment and Cleaning
            self.tprint_info("🧹 Step 1: Data quality assessment and cleaning")
            data_quality_result = self._assess_and_clean_data(market_data)
            
            if data_quality_result['quality_level'] == DataQualityLevel.CRITICAL:
                self.tprint_error("❌ Data quality is critical - processing aborted")
                return self._create_error_result("Critical data quality issues")
            
            cleaned_data = data_quality_result['cleaned_data']
            
            # Step 2: Generate Trading-Aware Labels
            self.tprint_info("🎯 Step 2: Generating trading-aware labels")
            label_result = self._generate_trading_aware_labels(
                cleaned_data, regime_data, portfolio_state
            )
            
            # Step 3: Label Stability Assessment
            self.tprint_info("🔍 Step 3: Assessing label stability")
            stability_result = self._assess_label_stability(
                label_result['labels'], cleaned_data
            )
            
            # Step 4: Apply Balancing and Weighting
            self.tprint_info("⚖️ Step 4: Applying balancing and weighting")
            balanced_result = self._apply_balancing_and_weighting(
                cleaned_data, label_result['labels'], label_result['confidence_scores']
            )
            
            # Step 5: Final Quality Check
            self.tprint_info("✅ Step 5: Final quality check")
            final_quality = self._perform_final_quality_check(
                balanced_result, stability_result, data_quality_result
            )
            
            # Compile results with standardized keys
            result = {
                'processed_data': balanced_result['X'],
                'labels': balanced_result['y'],
                'sample_weights': balanced_result['sample_weights'],
                'confidence_scores': label_result['confidence_scores'],
                'quality_metrics': {
                    'overall_score': data_quality_result.get('quality_score', 0.0),
                    'missing_rate': data_quality_result.get('quality_assessment', {}).get('missing_rate', 0.0),
                    'outlier_rate': data_quality_result.get('quality_assessment', {}).get('outlier_rate', 0.0),
                    'completeness': data_quality_result.get('quality_assessment', {}).get('completeness', 0.0),
                    'quality_level': data_quality_result.get('quality_level', DataQualityLevel.POOR).value,
                    'samples_removed': data_quality_result.get('samples_removed', 0),
                    'features_removed': data_quality_result.get('features_removed', 0)
                },
                'stability_metrics': {
                    'stability_score': stability_result.get('overall_stability', 0.0),
                    'label_drift': stability_result.get('drift_results', {}).get('drift_score', 0.0),
                    'autocorrelation': stability_result.get('autocorr_results', {}).get('autocorr_score', 0.0),
                    'leakage_detected': stability_result.get('leakage_results', {}).get('is_leakage_detected', False),
                    'stability_level': stability_result.get('stability_level', LabelStabilityLevel.UNSTABLE).value,
                    'recommendations': stability_result.get('recommendations', [])
                },
                'label_stats': label_result.get('label_stats', {}),
                'processing_time': time.time() - start_time,
                'original_samples': len(market_data),
                'processed_samples': len(balanced_result['X']),
                'timestamp': datetime.now(),
                'cache_key': cache_key
            }
            
            # Store in cache
            if self.config.enable_caching:
                self.cache[cache_key] = result
                self.cache_timestamps[cache_key] = datetime.now()
            
            # Update history
            self._update_history(result)
            
            self.tprint_success(f"✅ Enhanced data and labels processing completed in {result['processing_time']:.2f}s")
            self.tprint_info(f"   → Data quality: {data_quality_result['quality_level'].value}")
            self.tprint_info(f"   → Label stability: {stability_result['stability_level'].value}")
            self.tprint_info(f"   → Final quality: {final_quality['overall_score']:.3f}")
            
            return result
            
        except Exception as e:
            self.tprint_error(f"❌ Enhanced data and labels processing failed: {e}")
            return self._create_error_result(str(e))
    
    def _assess_and_clean_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality and apply comprehensive cleaning with robust error handling."""
        try:
            self.tprint_info("🔍 Assessing data quality...")
            
            # Validate input data
            is_valid, validation_msg = self._validate_input_data(market_data)
            if not is_valid:
                self.tprint_error(f"❌ Input validation failed: {validation_msg}")
                return self._create_data_quality_error_result(market_data, validation_msg)
            
            # Comprehensive data quality assessment with error handling
            try:
                quality_assessment = self.data_quality.calculate_data_quality_score(market_data)
            except Exception as e:
                self.tprint_warning(f"⚠️ Data quality assessment failed, using fallback: {e}")
                quality_assessment = self._fallback_quality_assessment(market_data)
            
            # Enhanced data cleaning with error handling
            try:
                cleaned_data, cleaning_report = self.data_quality.enhanced_automated_data_cleaning(
                    market_data, {
                        'missing_value_strategy': 'advanced_imputation',
                        'outlier_method': 'advanced_detection',
                        'correlation_threshold': 0.95,
                        'drift_adaptation': True,
                        'feature_stability_check': True
                    }
                )
            except Exception as e:
                self.tprint_warning(f"⚠️ Enhanced cleaning failed, using basic cleaning: {e}")
                cleaned_data = self._apply_basic_cleaning(market_data)
                cleaning_report = {'method': 'basic', 'error': str(e)}
            
            # Additional trading-specific cleaning
            try:
                cleaned_data = self._apply_trading_specific_cleaning(cleaned_data)
            except Exception as e:
                self.tprint_warning(f"⚠️ Trading-specific cleaning failed: {e}")
                # Continue with current cleaned_data
            
            # Calculate quality metrics
            quality_score = quality_assessment.get('overall_score', 0.0)
            missing_rate = quality_assessment.get('missing_rate', 0.0)
            outlier_rate = quality_assessment.get('outlier_rate', 0.0)
            completeness = quality_assessment.get('completeness', 0.0)
            
            # Determine quality level with robust thresholds
            if quality_score >= 0.9 and missing_rate < 0.05 and outlier_rate < 0.05:
                quality_level = DataQualityLevel.EXCELLENT
            elif quality_score >= 0.8 and missing_rate < 0.10 and outlier_rate < 0.10:
                quality_level = DataQualityLevel.GOOD
            elif quality_score >= 0.7 and missing_rate < 0.20 and outlier_rate < 0.20:
                quality_level = DataQualityLevel.FAIR
            elif quality_score >= 0.6 and missing_rate < 0.30 and outlier_rate < 0.30:
                quality_level = DataQualityLevel.POOR
            else:
                quality_level = DataQualityLevel.CRITICAL
            
            # Calculate effective sample size
            effective_sample_size = len(cleaned_data) * completeness
            
            result = {
                'original_data': market_data,
                'cleaned_data': cleaned_data,
                'quality_assessment': quality_assessment,
                'cleaning_report': cleaning_report,
                'quality_level': quality_level,
                'quality_score': quality_score,
                'missing_rate': missing_rate,
                'outlier_rate': outlier_rate,
                'completeness': completeness,
                'samples_removed': len(market_data) - len(cleaned_data),
                'features_removed': len(market_data.columns) - len(cleaned_data.columns),
                'effective_sample_size': effective_sample_size,
                'session_gap_anomalies': self._detect_session_gaps(cleaned_data),
                'dedup_count': self._count_duplicates(market_data, cleaned_data)
            }
            
            self.tprint_success(f"✅ Data cleaning completed: {quality_level.value} quality")
            self.tprint_info(f"   → Samples: {len(market_data)} → {len(cleaned_data)}")
            self.tprint_info(f"   → Features: {len(market_data.columns)} → {len(cleaned_data.columns)}")
            self.tprint_info(f"   → Quality score: {quality_score:.3f}")
            self.tprint_info(f"   → Effective samples: {effective_sample_size:.0f}")
            
            return result
            
        except Exception as e:
            self.tprint_error(f"❌ Data quality assessment failed: {e}")
            return self._create_data_quality_error_result(market_data, str(e))
    
    def _apply_trading_specific_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply trading-specific data cleaning rules with adaptive thresholds."""
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
            
            # Adaptive volume threshold based on data characteristics
            if 'volume' in cleaned.columns and len(cleaned) > 10:
                # Calculate adaptive volume threshold using 1st percentile of non-zero volumes
                non_zero_volumes = cleaned[cleaned['volume'] > 0]['volume']
                if len(non_zero_volumes) > 0:
                    min_volume_threshold = max(non_zero_volumes.quantile(0.01), 100.0)
                    zero_volume_mask = cleaned['volume'] < min_volume_threshold
                    cleaned = cleaned[~zero_volume_mask]
                else:
                    # Fallback: remove zero volumes
                    zero_volume_mask = cleaned['volume'] <= 0
                    cleaned = cleaned[~zero_volume_mask]
            
            # Adaptive extreme price change detection
            if len(cleaned) > 20:
                price_changes = cleaned['close'].pct_change().abs()
                
                # Calculate robust scale using median absolute deviation
                mad = np.median(np.abs(price_changes - np.median(price_changes)))
                robust_scale = mad * 1.4826  # Scale factor for normal distribution
                
                # Use 99.9th percentile of robust scale as threshold
                extreme_threshold = np.percentile(price_changes, 99.9)
                extreme_threshold = max(extreme_threshold, robust_scale * 5)  # At least 5 MAD
                
                extreme_change_mask = price_changes > extreme_threshold
                cleaned = cleaned[~extreme_change_mask]
                
                self.tprint_info(f"🔧 Adaptive price change threshold: {extreme_threshold:.4f}")
            
            # Ensure proper timestamp alignment
            if isinstance(cleaned.index, pd.DatetimeIndex):
                # Remove duplicate timestamps
                cleaned = cleaned[~cleaned.index.duplicated(keep='first')]
                
                # Sort by timestamp
                cleaned = cleaned.sort_index()
                
                # Check for session gaps and remove bars across major gaps
                time_diffs = cleaned.index.to_series().diff()
                median_interval = time_diffs.median()
                
                # Remove bars where gap is > 10x median interval (likely session gaps)
                large_gap_mask = time_diffs > (median_interval * 10)
                cleaned = cleaned[~large_gap_mask]
            
            return cleaned
            
        except Exception as e:
            self.tprint_warning(f"⚠️ Trading-specific cleaning failed: {e}")
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
            
            # Calculate volatility for regime conditioning using data-driven approach
            returns = market_data['close'].pct_change().dropna()
            
            # Use multiple volatility estimation methods and combine
            vol_rolling = returns.rolling(window=20).std() * np.sqrt(252)  # Rolling std
            vol_ewma = returns.ewm(span=20).std() * np.sqrt(252)  # EWMA
            vol_garch = self._estimate_garch_volatility(returns)  # GARCH approximation
            
            # Combine volatility estimates (weighted average)
            volatility = 0.5 * vol_rolling + 0.3 * vol_ewma + 0.2 * vol_garch
            
            # Align volatility with market_data index (fill NaN values at the beginning)
            volatility = volatility.reindex(market_data.index, fill_value=volatility.mean())
            
            # Data-driven regime detection using volatility percentiles
            if len(volatility.dropna()) > 50:
                vol_percentiles = volatility.quantile([0.25, 0.75])
                low_vol_threshold = vol_percentiles[0.25]
                high_vol_threshold = vol_percentiles[0.75]
                
                # Create regime labels
                regime_labels = pd.Series('normal', index=volatility.index)
                regime_labels[volatility <= low_vol_threshold] = 'low_vol'
                regime_labels[volatility >= high_vol_threshold] = 'high_vol'
                
                # Update regime_data if not provided
                if regime_data is None:
                    regime_data = regime_labels
            
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
            
            # Ensure regime_data is a categorical Series
            if regime_data is not None:
                if isinstance(regime_data, str):
                    regime_series = pd.Series(regime_data, index=market_data.index)
                else:
                    regime_series = regime_data.reindex(market_data.index, fill_value='unknown')
            else:
                regime_series = pd.Series('unknown', index=market_data.index)
            
            # Convert to categorical
            regime_series = pd.Categorical(regime_series, categories=regime_series.unique())
            
            # Create comprehensive labels DataFrame
            labels_df = pd.DataFrame({
                'analyst_label': analyst_labels,
                'analyst_confidence': analyst_confidence,
                'tactician_label': tactician_labels,
                'tactician_magnitude': tactician_magnitude,
                'volatility': volatility,
                'regime': regime_series
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
                                # Normalize by entropy to get NMI (0-1 range)
                                entropy_series = -np.sum(series.value_counts(normalize=True) * np.log2(series.value_counts(normalize=True) + 1e-8))
                                entropy_future = -np.sum(future_series.value_counts(normalize=True) * np.log2(future_series.value_counts(normalize=True) + 1e-8))
                                nmi_score = 2 * mi_score / (entropy_series + entropy_future + 1e-8)
                                leakage_scores.append(nmi_score)
                                leakage_details[f'{col}_mutual_info'] = mi_score
                                leakage_details[f'{col}_nmi'] = nmi_score
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
        """Detect label drift over time using data-driven thresholds."""
        try:
            drift_scores = []
            drift_details = {}
            
            # Use multiple time windows for drift detection
            n_samples = len(labels)
            if n_samples < 100:
                # For small datasets, use simple split
                split_point = n_samples // 2
                time_windows = [(0, split_point), (split_point, n_samples)]
            else:
                # For larger datasets, use rolling windows
                window_size = n_samples // 4
                time_windows = [
                    (0, window_size),  # Early period
                    (n_samples - window_size, n_samples)  # Late period
                ]
            
            for col in ['analyst_label', 'tactician_label']:
                if col in labels.columns:
                    col_drift_scores = []
                    
                    for start, end in time_windows:
                        period_labels = labels[col].iloc[start:end].dropna()
                        
                        if len(period_labels) > 10:
                            # Calculate multiple drift metrics
                            
                            # 1. Population Stability Index (PSI)
                            if len(col_drift_scores) > 0:
                                # Compare with previous period
                                prev_period = labels[col].iloc[time_windows[0][0]:time_windows[0][1]].dropna()
                                psi_score = self._calculate_psi(prev_period, period_labels)
                                col_drift_scores.append(psi_score)
                            
                            # 2. Jensen-Shannon Divergence
                            if len(col_drift_scores) > 0:
                                jsd_score = self._calculate_jsd(prev_period, period_labels)
                                col_drift_scores.append(jsd_score)
                            
                            # 3. Kolmogorov-Smirnov test
                            if len(col_drift_scores) > 0:
                                ks_stat, ks_pvalue = stats.ks_2samp(prev_period, period_labels)
                                col_drift_scores.append(ks_pvalue)
                            
                            prev_period = period_labels
                    
                    if col_drift_scores:
                        # Use minimum score (most conservative)
                        drift_score = min(col_drift_scores)
                        drift_scores.append(drift_score)
                        drift_details[f'{col}_drift_score'] = drift_score
                        drift_details[f'{col}_drift_methods'] = len(col_drift_scores)
            
            overall_drift = np.mean(drift_scores) if drift_scores else 1.0
            
            # Data-driven threshold based on historical baseline
            # In practice, this would be learned from historical data
            drift_threshold = 0.05  # 5% PSI threshold
            
            return {
                'drift_score': overall_drift,
                'drift_details': drift_details,
                'is_drift_detected': overall_drift < drift_threshold,
                'drift_threshold': drift_threshold
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
                            # Use Ljung-Box test for significance
                            try:
                                from statsmodels.stats.diagnostic import acorr_ljungbox
                                ljung_box_result = acorr_ljungbox(series, lags=min(5, len(series)//4), return_df=True)
                                p_value = ljung_box_result['lb_pvalue'].iloc[-1] if not ljung_box_result.empty else 0.5
                                # Higher p-value = less significant autocorrelation = better
                                autocorr_scores.append(p_value)
                            except:
                                # Fallback: use 1 - autocorr magnitude
                                autocorr_scores.append(1.0 - avg_autocorr)
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
        """Perform final quality check on all components with comprehensive validation."""
        try:
            self.tprint_info("✅ Performing final quality check...")
            
            # Calculate component scores with validation
            data_quality_score = self._validate_score(data_quality_result.get('quality_score', 0.0), 0.0, 1.0)
            label_stability_score = self._validate_score(stability_result.get('overall_stability', 0.0), 0.0, 1.0)
            
            # Calculate class balance using Hellinger distance
            class_dist = balanced_result.get('class_distribution', {})
            balance_score = self._calculate_balance_score(class_dist)
            
            # Calculate additional quality metrics
            effective_sample_size = balanced_result.get('balanced_size', 0)
            min_effective_samples = 100  # Minimum required samples
            
            # Calculate sample efficiency
            sample_efficiency = min(1.0, effective_sample_size / min_effective_samples) if min_effective_samples > 0 else 0.0
            
            # Calculate data completeness
            data_completeness = data_quality_result.get('completeness', 0.0)
            
            # Calculate stability components
            drift_score = stability_result.get('drift_results', {}).get('drift_score', 1.0)
            autocorr_score = stability_result.get('autocorr_results', {}).get('autocorr_score', 1.0)
            leakage_score = 1.0 - float(stability_result.get('leakage_results', {}).get('is_leakage_detected', False))
            
            # Calculate weighted component scores
            component_scores = {
                'data_quality': data_quality_score,
                'label_stability': label_stability_score,
                'class_balance': balance_score,
                'sample_efficiency': sample_efficiency,
                'data_completeness': data_completeness,
                'drift_resistance': drift_score,
                'autocorr_resistance': autocorr_score,
                'leakage_resistance': leakage_score
            }
            
            # Calculate overall score with learned weights
            weights = {
                'data_quality': 0.25,
                'label_stability': 0.20,
                'class_balance': 0.15,
                'sample_efficiency': 0.10,
                'data_completeness': 0.10,
                'drift_resistance': 0.08,
                'autocorr_resistance': 0.07,
                'leakage_resistance': 0.05
            }
            
            overall_score = sum(component_scores[key] * weights[key] for key in weights)
            overall_score = self._validate_score(overall_score, 0.0, 1.0)
            
            # Determine quality grade with more granular thresholds
            if overall_score >= 0.95:
                quality_grade = 'A+'
            elif overall_score >= 0.90:
                quality_grade = 'A'
            elif overall_score >= 0.85:
                quality_grade = 'A-'
            elif overall_score >= 0.80:
                quality_grade = 'B+'
            elif overall_score >= 0.75:
                quality_grade = 'B'
            elif overall_score >= 0.70:
                quality_grade = 'B-'
            elif overall_score >= 0.65:
                quality_grade = 'C+'
            elif overall_score >= 0.60:
                quality_grade = 'C'
            elif overall_score >= 0.55:
                quality_grade = 'C-'
            elif overall_score >= 0.50:
                quality_grade = 'D'
            else:
                quality_grade = 'F'
            
            # Generate comprehensive recommendations
            recommendations = self._generate_quality_recommendations(component_scores, weights)
            
            # Calculate quality flags
            quality_flags = {
                'data_quality_acceptable': data_quality_score >= 0.7,
                'stability_acceptable': label_stability_score >= 0.7,
                'balance_acceptable': balance_score >= 0.6,
                'sample_size_adequate': effective_sample_size >= min_effective_samples,
                'no_leakage_detected': not stability_result.get('leakage_results', {}).get('is_leakage_detected', False),
                'no_drift_detected': drift_score >= 0.8,
                'low_autocorr': autocorr_score >= 0.7
            }
            
            result = {
                'overall_score': overall_score,
                'quality_grade': quality_grade,
                'component_scores': component_scores,
                'component_weights': weights,
                'quality_flags': quality_flags,
                'recommendations': recommendations,
                'is_acceptable': overall_score >= self.config.min_data_quality_score,
                'effective_sample_size': effective_sample_size,
                'min_required_samples': min_effective_samples,
                'quality_trend': self._calculate_quality_trend(component_scores)
            }
            
            self.tprint_success(f"✅ Final quality check completed: {quality_grade} ({overall_score:.3f})")
            self.tprint_info(f"   → Data quality: {data_quality_score:.3f}")
            self.tprint_info(f"   → Label stability: {label_stability_score:.3f}")
            self.tprint_info(f"   → Class balance: {balance_score:.3f}")
            self.tprint_info(f"   → Effective samples: {effective_sample_size}")
            
            return result
            
        except Exception as e:
            self.tprint_error(f"❌ Final quality check failed: {e}")
            return {
                'overall_score': 0.0,
                'quality_grade': 'F',
                'component_scores': {},
                'quality_flags': {},
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
        """Generate cache key for the given inputs using content fingerprinting."""
        try:
            import hashlib
            
            # Create content fingerprint using hash of actual data
            data_fingerprint = hashlib.md5()
            
            # Hash index values
            data_fingerprint.update(market_data.index.values.tobytes())
            
            # Hash key columns (close, volume, etc.)
            key_cols = ['close', 'volume', 'open', 'high', 'low']
            for col in key_cols:
                if col in market_data.columns:
                    data_fingerprint.update(market_data[col].values.tobytes())
            
            # Hash regime data if available
            if regime_data is not None:
                data_fingerprint.update(str(regime_data.values).encode())
            
            # Hash portfolio state if available
            if portfolio_state is not None:
                data_fingerprint.update(str(sorted(portfolio_state.items())).encode())
            
            # Hash config (only key parameters that affect output)
            config_key = f"{self.config.min_data_quality_score}_{self.config.min_label_stability_score}_{self.config.enable_caching}"
            data_fingerprint.update(config_key.encode())
            
            return f"enhanced_labels_{data_fingerprint.hexdigest()[:16]}"
            
        except Exception as e:
            self.tprint_warning(f"⚠️ Cache key generation failed: {e}")
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
                'label_stats': result.get('label_stats', {}),
                'processing_time': result['processing_time']
            })
            
            # Update data quality history
            self.data_quality_history.append({
                'timestamp': result['timestamp'],
                'quality_level': result.get('quality_metrics', {}).get('quality_level', 'unknown'),
                'quality_score': result.get('quality_metrics', {}).get('overall_score', 0.0)
            })
            
            # Update stability history
            self.stability_history.append({
                'timestamp': result['timestamp'],
                'stability_level': result.get('stability_metrics', {}).get('stability_level', 'unknown'),
                'overall_stability': result.get('stability_metrics', {}).get('stability_score', 0.0)
            })
            
            # Keep only recent history (last 100 entries)
            for history in [self.label_history, self.data_quality_history, self.stability_history]:
                if len(history) > 100:
                    history[:] = history[-100:]
                    
        except Exception as e:
            tprint_warning(f"⚠️ Failed to update history: {e}")
    
    def _estimate_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """Estimate GARCH volatility using simplified approach."""
        try:
            # Simple GARCH(1,1) approximation
            # σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}
            
            # Initialize parameters (these would be estimated from data in practice)
            omega = 0.0001
            alpha = 0.1
            beta = 0.85
            
            # Initialize variance series
            variance = pd.Series(index=returns.index, dtype=float)
            variance.iloc[0] = returns.var()
            
            # Calculate GARCH variance
            for i in range(1, len(returns)):
                variance.iloc[i] = omega + alpha * (returns.iloc[i-1] ** 2) + beta * variance.iloc[i-1]
            
            # Convert to volatility (annualized)
            volatility = np.sqrt(variance) * np.sqrt(252)
            
            return volatility
            
        except Exception as e:
            self.tprint_warning(f"⚠️ GARCH volatility estimation failed: {e}")
            # Fallback to rolling standard deviation
            return returns.rolling(window=20).std() * np.sqrt(252)
    
    def _calculate_psi(self, expected: pd.Series, actual: pd.Series) -> float:
        """Calculate Population Stability Index (PSI)."""
        try:
            # Create bins for both series
            all_values = pd.concat([expected, actual])
            bins = pd.cut(all_values, bins=10, duplicates='drop')
            
            # Calculate expected and actual distributions
            expected_dist = expected.value_counts(bins=bins, normalize=True).sort_index()
            actual_dist = actual.value_counts(bins=bins, normalize=True).sort_index()
            
            # Align distributions
            common_bins = expected_dist.index.intersection(actual_dist.index)
            expected_dist = expected_dist.reindex(common_bins, fill_value=0)
            actual_dist = actual_dist.reindex(common_bins, fill_value=0)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-8
            expected_dist = expected_dist + epsilon
            actual_dist = actual_dist + epsilon
            
            # Calculate PSI
            psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
            
            return float(psi)
            
        except Exception as e:
            self.tprint_warning(f"⚠️ PSI calculation failed: {e}")
            return 0.0
    
    def _calculate_jsd(self, p: pd.Series, q: pd.Series) -> float:
        """Calculate Jensen-Shannon Divergence."""
        try:
            # Create bins for both series
            all_values = pd.concat([p, q])
            bins = pd.cut(all_values, bins=10, duplicates='drop')
            
            # Calculate distributions
            p_dist = p.value_counts(bins=bins, normalize=True).sort_index()
            q_dist = q.value_counts(bins=bins, normalize=True).sort_index()
            
            # Align distributions
            common_bins = p_dist.index.intersection(q_dist.index)
            p_dist = p_dist.reindex(common_bins, fill_value=0)
            q_dist = q_dist.reindex(common_bins, fill_value=0)
            
            # Normalize
            p_dist = p_dist / p_dist.sum()
            q_dist = q_dist / q_dist.sum()
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-8
            p_dist = p_dist + epsilon
            q_dist = q_dist + epsilon
            
            # Calculate JSD
            m = 0.5 * (p_dist + q_dist)
            jsd = 0.5 * stats.entropy(p_dist, m) + 0.5 * stats.entropy(q_dist, m)
            
            return float(jsd)
            
        except Exception as e:
            self.tprint_warning(f"⚠️ JSD calculation failed: {e}")
            return 0.0
    
    def _create_data_quality_error_result(self, market_data: pd.DataFrame, error_message: str) -> Dict[str, Any]:
        """Create data quality error result structure."""
        return {
            'original_data': market_data,
            'cleaned_data': market_data,
            'quality_assessment': {'overall_score': 0.0, 'missing_rate': 1.0, 'outlier_rate': 1.0, 'completeness': 0.0},
            'cleaning_report': {'method': 'error', 'error': error_message},
            'quality_level': DataQualityLevel.CRITICAL,
            'quality_score': 0.0,
            'missing_rate': 1.0,
            'outlier_rate': 1.0,
            'completeness': 0.0,
            'samples_removed': 0,
            'features_removed': 0,
            'effective_sample_size': 0,
            'session_gap_anomalies': 0,
            'dedup_count': 0,
            'error': error_message
        }
    
    def _fallback_quality_assessment(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback quality assessment when main assessment fails."""
        try:
            # Basic quality metrics
            missing_rate = market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns))
            
            # Simple outlier detection
            numeric_cols = market_data.select_dtypes(include=[np.number]).columns
            outlier_count = 0
            for col in numeric_cols:
                Q1 = market_data[col].quantile(0.25)
                Q3 = market_data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((market_data[col] < (Q1 - 1.5 * IQR)) | (market_data[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_count += outliers
            
            outlier_rate = outlier_count / (len(market_data) * len(numeric_cols)) if len(numeric_cols) > 0 else 0
            completeness = 1.0 - missing_rate
            
            # Overall score
            overall_score = (completeness + (1.0 - outlier_rate)) / 2
            
            return {
                'overall_score': overall_score,
                'missing_rate': missing_rate,
                'outlier_rate': outlier_rate,
                'completeness': completeness
            }
            
        except Exception:
            return {
                'overall_score': 0.0,
                'missing_rate': 1.0,
                'outlier_rate': 1.0,
                'completeness': 0.0
            }
    
    def _apply_basic_cleaning(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Apply basic data cleaning as fallback."""
        try:
            cleaned = market_data.copy()
            
            # Remove rows with all NaN values
            cleaned = cleaned.dropna(how='all')
            
            # Remove rows with negative prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in cleaned.columns:
                    cleaned = cleaned[cleaned[col] > 0]
            
            # Remove rows with zero volume
            if 'volume' in cleaned.columns:
                cleaned = cleaned[cleaned['volume'] > 0]
            
            return cleaned
            
        except Exception:
            return market_data
    
    def _detect_session_gaps(self, data: pd.DataFrame) -> int:
        """Detect session gaps in the data."""
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                return 0
            
            time_diffs = data.index.to_series().diff()
            median_interval = time_diffs.median()
            
            # Count gaps > 10x median interval
            large_gaps = (time_diffs > (median_interval * 10)).sum()
            return int(large_gaps)
            
        except Exception:
            return 0
    
    def _count_duplicates(self, original: pd.DataFrame, cleaned: pd.DataFrame) -> int:
        """Count duplicate rows removed during cleaning."""
        try:
            return len(original) - len(cleaned)
        except Exception:
            return 0
    
    def _validate_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Validate and clamp score to valid range."""
        try:
            if pd.isna(score) or not isinstance(score, (int, float)):
                return min_val
            return max(min_val, min(max_val, float(score)))
        except Exception:
            return min_val
    
    def _calculate_balance_score(self, class_dist: Dict) -> float:
        """Calculate class balance score using Hellinger distance."""
        try:
            if not class_dist or len(class_dist) < 2:
                return 0.5
            
            # Calculate class proportions
            total_samples = sum(class_dist.values())
            if total_samples == 0:
                return 0.0
            
            proportions = [count / total_samples for count in class_dist.values()]
            
            # Calculate Hellinger distance from uniform distribution
            uniform_prop = 1.0 / len(proportions)
            hellinger_dist = np.sqrt(sum((np.sqrt(p) - np.sqrt(uniform_prop)) ** 2 for p in proportions)) / np.sqrt(2)
            
            # Convert distance to score (closer to uniform = higher score)
            balance_score = 1.0 - hellinger_dist
            return self._validate_score(balance_score)
            
        except Exception:
            return 0.5
    
    def _generate_quality_recommendations(self, component_scores: Dict[str, float], weights: Dict[str, float]) -> List[str]:
        """Generate quality improvement recommendations based on component scores."""
        recommendations = []
        
        # Data quality recommendations
        if component_scores.get('data_quality', 0) < 0.7:
            recommendations.append("Improve data quality - address missing values and outliers")
        if component_scores.get('data_completeness', 0) < 0.8:
            recommendations.append("Increase data completeness - reduce missing data")
        
        # Stability recommendations
        if component_scores.get('label_stability', 0) < 0.7:
            recommendations.append("Address label stability issues - check for leakage and drift")
        if component_scores.get('drift_resistance', 0) < 0.8:
            recommendations.append("Reduce label drift - consider retraining or drift adaptation")
        if component_scores.get('leakage_resistance', 0) < 0.8:
            recommendations.append("Fix label leakage - review feature engineering and labeling logic")
        if component_scores.get('autocorr_resistance', 0) < 0.7:
            recommendations.append("Reduce autocorrelation - check for temporal dependencies")
        
        # Balance recommendations
        if component_scores.get('class_balance', 0) < 0.6:
            recommendations.append("Improve class balance - consider different balancing strategies")
        
        # Sample efficiency recommendations
        if component_scores.get('sample_efficiency', 0) < 0.8:
            recommendations.append("Increase sample size - collect more data or improve data quality")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Labels appear high quality - no immediate action required")
        
        return recommendations
    
    def _calculate_quality_trend(self, component_scores: Dict[str, float]) -> str:
        """Calculate quality trend based on component scores."""
        try:
            # Simple trend calculation based on score distribution
            scores = list(component_scores.values())
            if len(scores) < 2:
                return "stable"
            
            # Calculate weighted average of scores
            weights = [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05]
            if len(weights) >= len(scores):
                weights = weights[:len(scores)]
            else:
                weights = weights + [0.1] * (len(scores) - len(weights))
            
            weighted_avg = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            
            if weighted_avg >= 0.8:
                return "excellent"
            elif weighted_avg >= 0.7:
                return "good"
            elif weighted_avg >= 0.6:
                return "fair"
            else:
                return "poor"
                
        except Exception:
            return "unknown"
    
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
        """Get comprehensive system status and performance metrics."""
        try:
            status = {
                'system_initialized': True,
                'run_id': getattr(self, 'run_id', 'unknown'),
                'random_seed': getattr(self, 'random_seed', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'cache_status': {
                    'cache_size': len(self.cache),
                    'cache_enabled': self.config.enable_caching,
                    'cache_duration_hours': self.config.cache_duration_hours
                },
                'history_status': {
                    'labels': len(self.label_history),
                    'data_quality': len(self.data_quality_history),
                    'stability': len(self.stability_history),
                    'total_runs': max(len(self.label_history), len(self.data_quality_history), len(self.stability_history))
                },
                'configuration': {
                    'trading_objective': self.config.trading_objective.primary_objective,
                    'enable_caching': self.config.enable_caching,
                    'parallel_processing': self.config.parallel_processing,
                    'min_data_quality_score': self.config.min_data_quality_score,
                    'min_label_stability_score': self.config.min_label_stability_score,
                    'save_artifacts': self.config.save_artifacts
                },
                'monitoring': {
                    'thresholds_configured': hasattr(self, 'monitoring_config'),
                    'baseline_available': hasattr(self, 'baseline_metrics')
                }
            }
            
            # Add recent performance metrics
            if self.label_history:
                recent_labels = self.label_history[-1]
                status['recent_metrics'] = {
                    'processing_time': recent_labels.get('processing_time', 0.0),
                    'timestamp': recent_labels.get('timestamp', 'unknown')
                }
            
            if self.data_quality_history:
                recent_quality = self.data_quality_history[-1]
                status['recent_metrics']['quality_score'] = recent_quality.get('quality_score', 0.0)
            
            if self.stability_history:
                recent_stability = self.stability_history[-1]
                status['recent_metrics']['stability_score'] = recent_stability.get('overall_stability', 0.0)
            
            # Add performance trends
            if len(self.label_history) > 1:
                status['performance_trends'] = self._calculate_performance_trends()
            
            # Add system health assessment
            status['system_health'] = self._assess_system_health()
            
            return status
            
        except Exception as e:
            return {
                'system_initialized': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over recent runs."""
        try:
            if len(self.label_history) < 3:
                return {'trend_available': False}
            
            # Get recent processing times
            recent_times = [entry.get('processing_time', 0) for entry in self.label_history[-5:]]
            
            # Calculate trend
            if len(recent_times) >= 2:
                time_trend = 'improving' if recent_times[-1] < recent_times[0] else 'stable'
            else:
                time_trend = 'stable'
            
            return {
                'trend_available': True,
                'processing_time_trend': time_trend,
                'recent_avg_time': np.mean(recent_times),
                'time_volatility': np.std(recent_times)
            }
            
        except Exception:
            return {'trend_available': False}
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health."""
        try:
            health_score = 1.0
            issues = []
            
            # Check cache health
            if self.config.enable_caching and len(self.cache) == 0:
                health_score -= 0.1
                issues.append("Cache is empty despite being enabled")
            
            # Check history consistency
            history_lengths = [len(self.label_history), len(self.data_quality_history), len(self.stability_history)]
            if len(set(history_lengths)) > 1:
                health_score -= 0.1
                issues.append("History lengths are inconsistent across components")
            
            # Check recent performance
            if self.label_history:
                recent_time = self.label_history[-1].get('processing_time', 0)
                if recent_time > 300:  # 5 minutes
                    health_score -= 0.2
                    issues.append("Recent processing time is very high")
            
            # Determine health level
            if health_score >= 0.9:
                health_level = 'excellent'
            elif health_score >= 0.7:
                health_level = 'good'
            elif health_score >= 0.5:
                health_level = 'fair'
            else:
                health_level = 'poor'
            
            return {
                'health_score': health_score,
                'health_level': health_level,
                'issues': issues,
                'issue_count': len(issues)
            }
            
        except Exception:
            return {
                'health_score': 0.5,
                'health_level': 'unknown',
                'issues': ['Health assessment failed'],
                'issue_count': 1
            }
    
    def clear_cache(self):
        """Clear all cached results."""
        self.cache.clear()
        self.cache_timestamps.clear()
        self.tprint_info("🗑️ Cache cleared")
    
    def setup_monitoring_thresholds(self, reference_period_days: int = 90) -> Dict[str, Any]:
        """Setup data-driven monitoring thresholds based on historical baseline."""
        try:
            self.tprint_info("📊 Setting up data-driven monitoring thresholds...")
            
            # Calculate baseline metrics from historical data
            baseline_metrics = self._calculate_baseline_metrics(reference_period_days)
            
            # Set up adaptive thresholds
            monitoring_config = {
                'quality_thresholds': {
                    'warning': baseline_metrics['quality']['90th_percentile'],
                    'critical': baseline_metrics['quality']['95th_percentile'],
                    'excellent': baseline_metrics['quality']['75th_percentile']
                },
                'stability_thresholds': {
                    'warning': baseline_metrics['stability']['90th_percentile'],
                    'critical': baseline_metrics['stability']['95th_percentile'],
                    'excellent': baseline_metrics['stability']['75th_percentile']
                },
                'drift_thresholds': {
                    'warning': baseline_metrics['drift']['90th_percentile'],
                    'critical': baseline_metrics['drift']['95th_percentile']
                },
                'autocorr_thresholds': {
                    'warning': baseline_metrics['autocorr']['90th_percentile'],
                    'critical': baseline_metrics['autocorr']['95th_percentile']
                },
                'balance_thresholds': {
                    'warning': baseline_metrics['balance']['90th_percentile'],
                    'critical': baseline_metrics['balance']['95th_percentile']
                },
                'sample_size_thresholds': {
                    'warning': baseline_metrics['sample_size']['10th_percentile'],
                    'critical': baseline_metrics['sample_size']['5th_percentile']
                }
            }
            
            # Store monitoring config
            self.monitoring_config = monitoring_config
            self.baseline_metrics = baseline_metrics
            
            self.tprint_success("✅ Monitoring thresholds configured")
            self.tprint_info(f"   → Quality warning: {monitoring_config['quality_thresholds']['warning']:.3f}")
            self.tprint_info(f"   → Stability warning: {monitoring_config['stability_thresholds']['warning']:.3f}")
            self.tprint_info(f"   → Drift warning: {monitoring_config['drift_thresholds']['warning']:.3f}")
            
            return monitoring_config
            
        except Exception as e:
            self.tprint_error(f"❌ Monitoring setup failed: {e}")
            return self._get_default_monitoring_config()
    
    def _calculate_baseline_metrics(self, reference_period_days: int) -> Dict[str, Any]:
        """Calculate baseline metrics from historical data."""
        try:
            # Get historical data from the last reference_period_days
            cutoff_date = datetime.now() - timedelta(days=reference_period_days)
            
            # Filter history by date
            recent_quality = [entry for entry in self.data_quality_history 
                            if entry['timestamp'] >= cutoff_date]
            recent_stability = [entry for entry in self.stability_history 
                              if entry['timestamp'] >= cutoff_date]
            
            # Calculate percentiles for each metric
            quality_scores = [entry['quality_score'] for entry in recent_quality]
            stability_scores = [entry['overall_stability'] for entry in recent_stability]
            
            # Calculate additional metrics from recent runs
            drift_scores = []
            autocorr_scores = []
            balance_scores = []
            sample_sizes = []
            
            for entry in self.label_history:
                if entry['timestamp'] >= cutoff_date:
                    # Extract metrics from label_stats if available
                    label_stats = entry.get('label_stats', {})
                    if 'drift_score' in label_stats:
                        drift_scores.append(label_stats['drift_score'])
                    if 'autocorr_score' in label_stats:
                        autocorr_scores.append(label_stats['autocorr_score'])
                    if 'balance_score' in label_stats:
                        balance_scores.append(label_stats['balance_score'])
                    if 'processed_samples' in entry:
                        sample_sizes.append(entry['processed_samples'])
            
            # Calculate percentiles
            def safe_percentiles(data, percentiles=[5, 10, 25, 50, 75, 90, 95]):
                if not data:
                    return {f'{p}th_percentile': 0.5 for p in percentiles}
                return {f'{p}th_percentile': np.percentile(data, p) for p in percentiles}
            
            return {
                'quality': safe_percentiles(quality_scores),
                'stability': safe_percentiles(stability_scores),
                'drift': safe_percentiles(drift_scores),
                'autocorr': safe_percentiles(autocorr_scores),
                'balance': safe_percentiles(balance_scores),
                'sample_size': safe_percentiles(sample_sizes),
                'reference_period_days': reference_period_days,
                'total_samples': len(quality_scores)
            }
            
        except Exception as e:
            self.tprint_warning(f"⚠️ Baseline calculation failed: {e}")
            return self._get_default_baseline_metrics()
    
    def _get_default_monitoring_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration when baseline calculation fails."""
        return {
            'quality_thresholds': {'warning': 0.7, 'critical': 0.5, 'excellent': 0.8},
            'stability_thresholds': {'warning': 0.6, 'critical': 0.4, 'excellent': 0.8},
            'drift_thresholds': {'warning': 0.1, 'critical': 0.2},
            'autocorr_thresholds': {'warning': 0.3, 'critical': 0.5},
            'balance_thresholds': {'warning': 0.6, 'critical': 0.4},
            'sample_size_thresholds': {'warning': 1000, 'critical': 500}
        }
    
    def _get_default_baseline_metrics(self) -> Dict[str, Any]:
        """Get default baseline metrics when calculation fails."""
        return {
            'quality': {f'{p}th_percentile': 0.7 for p in [5, 10, 25, 50, 75, 90, 95]},
            'stability': {f'{p}th_percentile': 0.6 for p in [5, 10, 25, 50, 75, 90, 95]},
            'drift': {f'{p}th_percentile': 0.1 for p in [5, 10, 25, 50, 75, 90, 95]},
            'autocorr': {f'{p}th_percentile': 0.3 for p in [5, 10, 25, 50, 75, 90, 95]},
            'balance': {f'{p}th_percentile': 0.6 for p in [5, 10, 25, 50, 75, 90, 95]},
            'sample_size': {f'{p}th_percentile': 1000 for p in [5, 10, 25, 50, 75, 90, 95]},
            'reference_period_days': 90,
            'total_samples': 0
        }
    
    def check_monitoring_alerts(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Check for monitoring alerts based on current result."""
        try:
            if not hasattr(self, 'monitoring_config'):
                self.setup_monitoring_thresholds()
            
            alerts = {
                'quality_alerts': [],
                'stability_alerts': [],
                'drift_alerts': [],
                'autocorr_alerts': [],
                'balance_alerts': [],
                'sample_size_alerts': [],
                'overall_status': 'normal',
                'alert_count': 0
            }
            
            # Check quality alerts
            quality_score = result.get('quality_metrics', {}).get('overall_score', 0.0)
            if quality_score < self.monitoring_config['quality_thresholds']['critical']:
                alerts['quality_alerts'].append(f"CRITICAL: Quality score {quality_score:.3f} below critical threshold")
                alerts['overall_status'] = 'critical'
            elif quality_score < self.monitoring_config['quality_thresholds']['warning']:
                alerts['quality_alerts'].append(f"WARNING: Quality score {quality_score:.3f} below warning threshold")
                if alerts['overall_status'] == 'normal':
                    alerts['overall_status'] = 'warning'
            
            # Check stability alerts
            stability_score = result.get('stability_metrics', {}).get('stability_score', 0.0)
            if stability_score < self.monitoring_config['stability_thresholds']['critical']:
                alerts['stability_alerts'].append(f"CRITICAL: Stability score {stability_score:.3f} below critical threshold")
                alerts['overall_status'] = 'critical'
            elif stability_score < self.monitoring_config['stability_thresholds']['warning']:
                alerts['stability_alerts'].append(f"WARNING: Stability score {stability_score:.3f} below warning threshold")
                if alerts['overall_status'] == 'normal':
                    alerts['overall_status'] = 'warning'
            
            # Check drift alerts
            drift_score = result.get('stability_metrics', {}).get('label_drift', 0.0)
            if drift_score > self.monitoring_config['drift_thresholds']['critical']:
                alerts['drift_alerts'].append(f"CRITICAL: Drift score {drift_score:.3f} above critical threshold")
                alerts['overall_status'] = 'critical'
            elif drift_score > self.monitoring_config['drift_thresholds']['warning']:
                alerts['drift_alerts'].append(f"WARNING: Drift score {drift_score:.3f} above warning threshold")
                if alerts['overall_status'] == 'normal':
                    alerts['overall_status'] = 'warning'
            
            # Check autocorrelation alerts
            autocorr_score = result.get('stability_metrics', {}).get('autocorrelation', 0.0)
            if autocorr_score > self.monitoring_config['autocorr_thresholds']['critical']:
                alerts['autocorr_alerts'].append(f"CRITICAL: Autocorrelation {autocorr_score:.3f} above critical threshold")
                alerts['overall_status'] = 'critical'
            elif autocorr_score > self.monitoring_config['autocorr_thresholds']['warning']:
                alerts['autocorr_alerts'].append(f"WARNING: Autocorrelation {autocorr_score:.3f} above warning threshold")
                if alerts['overall_status'] == 'normal':
                    alerts['overall_status'] = 'warning'
            
            # Check sample size alerts
            processed_samples = result.get('processed_samples', 0)
            if processed_samples < self.monitoring_config['sample_size_thresholds']['critical']:
                alerts['sample_size_alerts'].append(f"CRITICAL: Sample size {processed_samples} below critical threshold")
                alerts['overall_status'] = 'critical'
            elif processed_samples < self.monitoring_config['sample_size_thresholds']['warning']:
                alerts['sample_size_alerts'].append(f"WARNING: Sample size {processed_samples} below warning threshold")
                if alerts['overall_status'] == 'normal':
                    alerts['overall_status'] = 'warning'
            
            # Count total alerts
            alerts['alert_count'] = sum(len(alert_list) for alert_list in alerts.values() 
                                      if isinstance(alert_list, list))
            
            # Log alerts
            if alerts['alert_count'] > 0:
                self.tprint_warning(f"⚠️ {alerts['alert_count']} monitoring alerts detected")
                for alert_type, alert_list in alerts.items():
                    if isinstance(alert_list, list) and alert_list:
                        for alert in alert_list:
                            self.tprint_warning(f"   → {alert}")
            else:
                self.tprint_success("✅ No monitoring alerts - all metrics within normal ranges")
            
            return alerts
            
        except Exception as e:
            self.tprint_error(f"❌ Monitoring check failed: {e}")
            return {
                'quality_alerts': [],
                'stability_alerts': [],
                'drift_alerts': [],
                'autocorr_alerts': [],
                'balance_alerts': [],
                'sample_size_alerts': [],
                'overall_status': 'error',
                'alert_count': 0,
                'error': str(e)
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all processing runs."""
        try:
            if not self.label_history:
                return {'message': 'No processing history available'}
            
            # Calculate performance metrics
            processing_times = [entry['processing_time'] for entry in self.label_history]
            quality_scores = [entry['quality_score'] for entry in self.data_quality_history]
            stability_scores = [entry['overall_stability'] for entry in self.stability_history]
            
            # Calculate trends using robust slope estimation
            def robust_slope(data):
                if len(data) < 3:
                    return 0.0
                x = np.arange(len(data))
                # Use Theil-Sen estimator for robust slope
                slopes = []
                for i in range(len(data)):
                    for j in range(i + 1, len(data)):
                        if x[j] != x[i]:
                            slopes.append((data[j] - data[i]) / (x[j] - x[i]))
                return np.median(slopes) if slopes else 0.0
            
            summary = {
                'total_runs': len(self.label_history),
                'avg_processing_time': np.mean(processing_times),
                'avg_quality_score': np.mean(quality_scores) if quality_scores else 0.0,
                'avg_stability_score': np.mean(stability_scores) if stability_scores else 0.0,
                'cache_hit_rate': len(self.cache) / max(1, len(self.label_history)),
                'recent_trends': {
                    'processing_time_trend': 'improving' if robust_slope(processing_times) < 0 else 'stable',
                    'quality_trend': 'improving' if robust_slope(quality_scores) > 0 else 'stable',
                    'stability_trend': 'improving' if robust_slope(stability_scores) > 0 else 'stable'
                }
            }
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}
    
    def create_purged_cv_splits(self, data: pd.DataFrame, n_splits: int = 5, 
                               purge_days: int = 1, embargo_days: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create purged and embargoed cross-validation splits for time series data."""
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Data must have DatetimeIndex for purged CV")
            
            # Convert days to appropriate time units
            purge_td = pd.Timedelta(days=purge_days)
            embargo_td = pd.Timedelta(days=embargo_days)
            
            splits = []
            n_samples = len(data)
            
            # Create time-based splits
            for i in range(n_splits):
                # Calculate split boundaries
                start_idx = int(i * n_samples / n_splits)
                end_idx = int((i + 1) * n_samples / n_splits)
                
                # Get time boundaries
                start_time = data.index[start_idx]
                end_time = data.index[end_idx - 1]
                
                # Create purged and embargoed boundaries
                purge_start = end_time + embargo_td
                purge_end = purge_start + purge_td
                
                # Training set: before purge period
                train_mask = data.index < purge_start
                train_indices = np.where(train_mask)[0]
                
                # Test set: after purge period
                test_mask = data.index >= purge_end
                test_indices = np.where(test_mask)[0]
                
                if len(train_indices) > 0 and len(test_indices) > 0:
                    splits.append((train_indices, test_indices))
            
            return splits
            
        except Exception as e:
            self.tprint_error(f"❌ Purged CV creation failed: {e}")
            # Fallback to simple time series split
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=n_splits)
            return list(tscv.split(data))
    
    def generate_label_manifest(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate label manifest for reproducibility and traceability."""
        try:
            import hashlib
            
            manifest = {
                'timestamp': result.get('timestamp', datetime.now()).isoformat(),
                'data_checksum': self._calculate_data_checksum(result.get('processed_data', pd.DataFrame())),
                'label_checksum': self._calculate_data_checksum(result.get('labels', pd.DataFrame())),
                'config_hash': hashlib.md5(str(self.config).encode()).hexdigest()[:16],
                'processing_time': result.get('processing_time', 0.0),
                'original_samples': result.get('original_samples', 0),
                'processed_samples': result.get('processed_samples', 0),
                'quality_score': result.get('quality_metrics', {}).get('overall_score', 0.0),
                'stability_score': result.get('stability_metrics', {}).get('stability_score', 0.0),
                'version': '1.0.0'  # System version
            }
            
            return manifest
            
        except Exception as e:
            self.tprint_warning(f"⚠️ Label manifest generation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_data_checksum(self, data: Union[pd.DataFrame, pd.Series]) -> str:
        """Calculate checksum for data integrity verification."""
        try:
            import hashlib
            
            if isinstance(data, pd.Series):
                data = data.to_frame()
            
            # Create hash of data content
            checksum = hashlib.md5()
            checksum.update(data.index.values.tobytes())
            checksum.update(data.values.tobytes())
            
            return checksum.hexdigest()[:16]
            
        except Exception:
            return "unknown"
    
    def _serialize_enums(self, data: Any) -> Any:
        """Recursively serialize enums to strings for JSON compatibility."""
        if isinstance(data, Enum):
            return data.value
        elif isinstance(data, dict):
            return {key: self._serialize_enums(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._serialize_enums(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._serialize_enums(item) for item in data)
        else:
            return data
    
    def _validate_input_data(self, market_data: pd.DataFrame) -> Tuple[bool, str]:
        """Validate input market data format and content."""
        try:
            # Check if DataFrame
            if not isinstance(market_data, pd.DataFrame):
                return False, "market_data must be a pandas DataFrame"
            
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in market_data.columns]
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
            
            # Check data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if not pd.api.types.is_numeric_dtype(market_data[col]):
                    return False, f"Column {col} must be numeric"
            
            # Check for empty DataFrame
            if len(market_data) == 0:
                return False, "market_data cannot be empty"
            
            # Check for all NaN values
            if market_data.isnull().all().all():
                return False, "market_data contains only NaN values"
            
            # Check price validity
            price_cols = ['open', 'high', 'low', 'close']
            if (market_data[price_cols] <= 0).any().any():
                return False, "Price columns must contain positive values"
            
            # Check OHLC consistency
            if not ((market_data['high'] >= market_data['low']) & 
                   (market_data['high'] >= market_data['open']) & 
                   (market_data['high'] >= market_data['close']) &
                   (market_data['low'] <= market_data['open']) & 
                   (market_data['low'] <= market_data['close'])).all():
                return False, "OHLC data is inconsistent (high < low, etc.)"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _ensure_result_completeness(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure result has all required fields with proper defaults."""
        try:
            # Define required fields with defaults
            required_fields = {
                'processed_data': pd.DataFrame(),
                'labels': pd.DataFrame(),
                'sample_weights': pd.Series(dtype=float),
                'quality_metrics': {},
                'stability_metrics': {},
                'label_stats': {},
                'processing_time': 0.0,
                'original_samples': 0,
                'processed_samples': 0,
                'timestamp': datetime.now(),
                'cache_key': '',
                'success': True
            }
            
            # Add missing fields
            for field, default_value in required_fields.items():
                if field not in result:
                    result[field] = default_value
            
            # Ensure enums are serialized
            result = self._serialize_enums(result)
            
            return result
            
        except Exception as e:
            self.tprint_warning(f"⚠️ Result completeness check failed: {e}")
            return result


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