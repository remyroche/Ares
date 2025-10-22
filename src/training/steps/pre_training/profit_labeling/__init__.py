"""
Volatility-Aware Multi-Horizon Profit Labeling System

This module provides a comprehensive, data-driven profit labeling system that explicitly
accounts for volatility and microstructure noise, optimized for creating strong labels
that are learnable by ML models and generalize well.

Key Features:
- Volatility-normalized target bands and horizons
- Event-based bar construction with microstructure filtering
- Noise gating and eligibility filters
- Multi-target scheme (small/medium/high) with data-driven selection
- Label quality scoring and optimization
- Integration with existing utility modules

Components:
- volatility_aware_labeler: Main labeling system
- bar_construction: Event-based bar construction utilities
- volatility_modeling: Volatility estimation and normalization
- noise_gating: Microstructure noise filtering
- quality_scoring: Label quality assessment and optimization
- multi_target_scheme: Multi-target labeling with data-driven selection
"""

from .volatility_aware_profit_labeler import (
    VolatilityAwareProfitLabeler as VolatilityAwareMultiHorizonLabeler,
    VolatilityAwareConfig,
    LabelQualityMetrics as LabelQualityScore,
    LabelingResult
)

from .bar_construction import (
    EventBasedBarConstructor,
    BarConstructionConfig,
    BarConstructionResult
)

from .volatility_modeling import (
    VolatilityModeler,
    VolatilityConfig,
    VolatilityResult
)

from .noise_gating import (
    NoiseGatingFilter,
    NoiseGatingConfig,
    EligibilityResult
)

from .quality_scoring import (
    LabelQualityScorer,
    QualityScoringConfig,
    QualityMetrics
)

from .multi_target_scheme import (
    MultiTargetScheme,
    MultiTargetConfig,
    TargetSelectionResult
)

from .profit_labeling_report_generator import (
    ProfitLabelingReportGenerator,
    generate_profit_labeling_report,
    ProfitLabelingReport
)

__version__ = "1.0.0"
__author__ = "Ares Trading System"

__all__ = [
    # Main labeler
    "VolatilityAwareMultiHorizonLabeler",
    "VolatilityAwareConfig", 
    "LabelQualityScore",
    "LabelingResult",
    
    # Bar construction
    "EventBasedBarConstructor",
    "BarConstructionConfig",
    "BarConstructionResult",
    
    # Volatility modeling
    "VolatilityModeler",
    "VolatilityConfig", 
    "VolatilityResult",
    
    # Noise gating
    "NoiseGatingFilter",
    "NoiseGatingConfig",
    "EligibilityResult",
    
    # Quality scoring
    "LabelQualityScorer",
    "QualityScoringConfig",
    "QualityMetrics",
    
    # Multi-target scheme
    "MultiTargetScheme",
    "MultiTargetConfig",
    "TargetSelectionResult",

    # Report generation
    "ProfitLabelingReportGenerator",
    "generate_profit_labeling_report",
    "ProfitLabelingReport",
]