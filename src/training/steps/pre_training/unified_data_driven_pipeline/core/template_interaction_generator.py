"""
Template-Based Interaction Generation Component

This module provides template-based interaction generation inspired by HTFInteractionTemplates,
with core 15 interaction templates and HTF-aware templates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import time
from collections import defaultdict
from itertools import combinations, product
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
        tprint_debug, tprint_performance
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

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

logger = logging.getLogger(__name__)

@dataclass
class FeatureSpec:
    """Standardized feature specification with metadata."""
    name: str
    series: pd.Series
    role: str                         # e.g. "price_feature", "volatility_feature"
    tags: List[str] = field(default_factory=list)  # e.g. ["close","spot"], ["rv_5m"]
    freq: Optional[str] = None
    window: Optional[int] = None
    scale: Optional[str] = None       # e.g. "zscore", "minmax", None
    preprocess: Dict[str, Any] = field(default_factory=dict)  # e.g. {"winsor": (0.01, 0.99)}

class FeatureStore:
    """Standardizes, aligns, and serves Series by semantic role."""
    
    def __init__(self, features: Union[pd.DataFrame, Dict[str, pd.Series], List[FeatureSpec]]):
        self._raw = {}
        if isinstance(features, pd.DataFrame):
            for col in features.columns:
                self._raw[col] = FeatureSpec(name=col, series=features[col], role="unknown")
        elif isinstance(features, dict):
            for k, v in features.items():
                self._raw[k] = FeatureSpec(name=k, series=v, role="unknown")
        elif isinstance(features, list):
            for fs in features:
                self._raw[fs.name] = fs
        else:
            self._raw = {}

        self._index = self._infer_common_index()
        self._registry_by_role: Dict[str, List[str]] = {}
        self._cache: Dict[str, pd.Series] = {}
        self._op_cache: Dict[str, pd.Series] = {}  # Cache for operations

        # sanitize, align, enforce float dtype
        for k, fs in self._raw.items():
            s = fs.series
            if not s.index.equals(self._index):
                s = s.reindex(self._index)
            s = s.astype("float64")
            # simple preprocessing hooks
            wins = fs.preprocess.get("winsor")
            if wins is not None:
                lo, hi = wins
                s = s.clip(s.quantile(lo), s.quantile(hi))
            if fs.scale == "zscore":
                s = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
            self._raw[k].series = s

        # default role inference (keeps your heuristics)
        for k, fs in self._raw.items():
            if fs.role == "unknown":
                fs.role = self._infer_role_from_name(fs.name)

            self._registry_by_role.setdefault(fs.role, []).append(k)

    def _infer_common_index(self) -> pd.Index:
        idxs = [fs.series.index for fs in self._raw.values() if isinstance(fs.series, pd.Series)]
        if not idxs:
            return pd.RangeIndex(0, 100)  # fallback
        # choose the longest index and reindex others to it
        return max(idxs, key=len)

    @staticmethod
    def _infer_role_from_name(name: str) -> str:
        n = name.lower()
        if any(x in n for x in ['price', 'close', 'open', 'high', 'low']): return 'price_feature'
        if any(x in n for x in ['vol', 'sigma', 'rv', 'gk']): return 'volatility_feature'
        if any(x in n for x in ['mom', 'momentum', 'signal', 'alpha']): return 'momentum_feature'
        if any(x in n for x in ['rsi', 'stoch', 'mean_rev', 'osc']): return 'mean_reversion_feature'
        if any(x in n for x in ['liquidity', 'depth', 'book']): return 'liquidity_feature'
        if 'volume' in n: return 'volume_feature'
        if any(x in n for x in ['tod', 'time_of_day', 'session']): return 'tod_indicator'
        if any(x in n for x in ['regime', 'vol_regime']): return 'regime_indicator'
        return 'feature'  # generic

    @property
    def index(self) -> pd.Index:
        return self._index

    def by_role(self, role: str) -> List[str]:
        return self._registry_by_role.get(role, [])

    def get(self, name: str) -> pd.Series:
        if name in self._cache:
            return self._cache[name]
        s = self._raw[name].series
        # final NaN policy: forward-fill then drop leading
        s = s.ffill()
        self._cache[name] = s
        return s

    def get_cached_op(self, op_key: str) -> Optional[pd.Series]:
        """Get cached operation result."""
        return self._op_cache.get(op_key)

    def cache_op(self, op_key: str, result: pd.Series) -> None:
        """Cache operation result."""
        self._op_cache[op_key] = result

    def names(self) -> List[str]:
        return list(self._raw.keys())

@dataclass
class InteractionTemplate:
    """Template for generating interactions."""
    name: str
    template_type: str  # 'core', 'htf_aware'
    formula: str
    required_features: List[str]
    optional_features: List[str]
    max_instances: int
    priority: int
    metadata: Dict[str, Any]

@dataclass
class GeneratedInteraction:
    """Generated interaction feature."""
    name: str
    formula: str
    parent_features: List[str]
    interaction_type: str
    feature_series: pd.Series
    utility_score: float
    metadata: Dict[str, Any]

@dataclass
class TemplateConfig:
    """Configuration for template-based interaction generation."""

    # Budget allocation
    total_budget: int = 30
    core_budget: int = 15
    htf_aware_budget: int = 15

    # Quality thresholds
    min_utility_score: float = 0.1
    max_correlation_threshold: float = 0.95

    # Performance settings
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    memory_efficient: bool = True
    random_state: Optional[int] = None

    # Template settings
    enable_core_templates: bool = True
    enable_htf_templates: bool = True
    enable_interaction_heredity: bool = True

class CoreInteractionTemplates:
    """Core 15 interaction templates (theory-first)."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = self._create_core_templates()
        tprint_info(f"Initialized core interaction templates | count={len(self.templates)}")

    def _create_core_templates(self) -> List[InteractionTemplate]:
        """Create core interaction templates."""
        templates = [
            # Price-Volatility interactions
            InteractionTemplate(
                name="price_vol_interaction",
                template_type="core",
                formula="price_feature * volatility_feature",
                required_features=["price_feature", "volatility_feature"],
                optional_features=[],
                max_instances=5,
                priority=1,
                metadata={"description": "Price-volatility interaction"}
            ),

            # Momentum-Mean Reversion interactions
            InteractionTemplate(
                name="momentum_meanrev_interaction",
                template_type="core",
                formula="momentum_feature * mean_reversion_feature",
                required_features=["momentum_feature", "mean_reversion_feature"],
                optional_features=[],
                max_instances=5,
                priority=1,
                metadata={"description": "Momentum-mean reversion interaction"}
            ),

            # Liquidity-Price interactions
            InteractionTemplate(
                name="liquidity_price_interaction",
                template_type="core",
                formula="liquidity_feature * price_feature",
                required_features=["liquidity_feature", "price_feature"],
                optional_features=[],
                max_instances=5,
                priority=1,
                metadata={"description": "Liquidity-price interaction"}
            ),

            # Volatility-Volume interactions
            InteractionTemplate(
                name="vol_volume_interaction",
                template_type="core",
                formula="volatility_feature * volume_feature",
                required_features=["volatility_feature", "volume_feature"],
                optional_features=[],
                max_instances=5,
                priority=1,
                metadata={"description": "Volatility-volume interaction"}
            ),

            # Time-of-day interactions
            InteractionTemplate(
                name="tod_interaction",
                template_type="core",
                formula="feature * tod_indicator",
                required_features=["feature", "tod_indicator"],
                optional_features=[],
                max_instances=3,
                priority=2,
                metadata={"description": "Time-of-day interaction"}
            ),

            # Cross-sectional interactions
            InteractionTemplate(
                name="cross_sectional_interaction",
                template_type="core",
                formula="feature - market_feature",
                required_features=["feature", "market_feature"],
                optional_features=[],
                max_instances=3,
                priority=2,
                metadata={"description": "Cross-sectional interaction"}
            ),

            # Regime interactions
            InteractionTemplate(
                name="regime_interaction",
                template_type="core",
                formula="feature * regime_indicator",
                required_features=["feature", "regime_indicator"],
                optional_features=[],
                max_instances=3,
                priority=2,
                metadata={"description": "Regime interaction"}
            ),

            # Lag interactions
            InteractionTemplate(
                name="lag_interaction",
                template_type="core",
                formula="feature * feature_lag",
                required_features=["feature", "feature_lag"],
                optional_features=[],
                max_instances=5,
                priority=3,
                metadata={"description": "Lag interaction"}
            ),

            # Polynomial interactions
            InteractionTemplate(
                name="polynomial_interaction",
                template_type="core",
                formula="feature ** 2",
                required_features=["feature"],
                optional_features=[],
                max_instances=3,
                priority=3,
                metadata={"description": "Polynomial interaction"}
            ),

            # Ratio interactions
            InteractionTemplate(
                name="ratio_interaction",
                template_type="core",
                formula="feature1 / (feature2 + epsilon)",
                required_features=["feature1", "feature2"],
                optional_features=[],
                max_instances=5,
                priority=3,
                metadata={"description": "Ratio interaction", "defaults": {"epsilon": 1e-6}}
            ),

            # Difference interactions
            InteractionTemplate(
                name="difference_interaction",
                template_type="core",
                formula="feature1 - feature2",
                required_features=["feature1", "feature2"],
                optional_features=[],
                max_instances=5,
                priority=3,
                metadata={"description": "Difference interaction"}
            ),

            # Product interactions
            InteractionTemplate(
                name="product_interaction",
                template_type="core",
                formula="feature1 * feature2",
                required_features=["feature1", "feature2"],
                optional_features=[],
                max_instances=5,
                priority=3,
                metadata={"description": "Product interaction"}
            ),

            # Conditional interactions
            InteractionTemplate(
                name="conditional_interaction",
                template_type="core",
                formula="feature * (condition > threshold)",
                required_features=["feature", "condition"],
                optional_features=["threshold"],
                max_instances=3,
                priority=3,
                metadata={"description": "Conditional interaction", "defaults": {"threshold": 0.0}}
            ),

            # Rolling interactions
            InteractionTemplate(
                name="rolling_interaction",
                template_type="core",
                formula="feature.rolling(window).mean()",
                required_features=["feature"],
                optional_features=["window"],
                max_instances=3,
                priority=3,
                metadata={"description": "Rolling interaction", "defaults": {"window": 20}}
            ),

            # Z-score interactions
            InteractionTemplate(
                name="zscore_interaction",
                template_type="core",
                formula="(feature - feature.mean()) / feature.std()",
                required_features=["feature"],
                optional_features=[],
                max_instances=3,
                priority=3,
                metadata={"description": "Z-score interaction"}
            )
        ]

        return templates

class HTFAwareTemplates:
    """HTF-aware interaction templates."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = self._create_htf_aware_templates()
        tprint_info(f"Initialized HTF-aware templates | count={len(self.templates)}")

    def _create_htf_aware_templates(self) -> List[InteractionTemplate]:
        """Create HTF-aware interaction templates."""
        templates = [
            # HTF trend × base-TF liquidity
            InteractionTemplate(
                name="htf_trend_liquidity_interaction",
                template_type="htf_aware",
                formula="htf_trend_feature * base_liquidity_feature",
                required_features=["htf_trend_feature", "base_liquidity_feature"],
                optional_features=[],
                max_instances=3,
                priority=1,
                metadata={"description": "HTF trend × base liquidity interaction"}
            ),

            # HTF vol × base signal strength
            InteractionTemplate(
                name="htf_vol_signal_interaction",
                template_type="htf_aware",
                formula="htf_volatility_feature * base_signal_feature",
                required_features=["htf_volatility_feature", "base_signal_feature"],
                optional_features=[],
                max_instances=3,
                priority=1,
                metadata={"description": "HTF volatility × base signal interaction"}
            ),

            # HTF momentum conflict
            InteractionTemplate(
                name="htf_momentum_conflict_interaction",
                template_type="htf_aware",
                formula="htf_momentum_feature * (-base_momentum_feature)",
                required_features=["htf_momentum_feature", "base_momentum_feature"],
                optional_features=[],
                max_instances=3,
                priority=1,
                metadata={"description": "HTF momentum conflict interaction"}
            ),

            # HTF regime × base feature
            InteractionTemplate(
                name="htf_regime_base_interaction",
                template_type="htf_aware",
                formula="htf_regime_feature * base_feature",
                required_features=["htf_regime_feature", "base_feature"],
                optional_features=[],
                max_instances=3,
                priority=2,
                metadata={"description": "HTF regime × base feature interaction"}
            ),

            # HTF anchor × base deviation
            InteractionTemplate(
                name="htf_anchor_deviation_interaction",
                template_type="htf_aware",
                formula="htf_anchor_feature * base_deviation_feature",
                required_features=["htf_anchor_feature", "base_deviation_feature"],
                optional_features=[],
                max_instances=3,
                priority=2,
                metadata={"description": "HTF anchor × base deviation interaction"}
            )
        ]

        return templates

class TemplateInteractionGenerator:
    """Template-based interaction generator with VectorBT optimization."""

    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        Initialize the template interaction generator.

        Args:
            config: Configuration for template generation
        """
        self.config = config or TemplateConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize template systems
        self.core_templates = CoreInteractionTemplates()
        self.htf_aware_templates = HTFAwareTemplates()

        # Initialize formula operations
        self.formula_ops = self._build_formula_ops()

        # Performance tracking
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_generation_time': 0.0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'interactions_generated': 0
        }

        tprint_info("🎯 Template Interaction Generator initialized")
        tprint_debug(f"📊 Core templates: {len(self.core_templates.templates)}")
        tprint_debug(f"📊 HTF templates: {len(self.htf_aware_templates.templates)}")
        tprint_debug(f"📊 VectorBT available: {VECTORBT_AVAILABLE}")

    def _build_formula_ops(self) -> Dict[str, Any]:
        """Build formula operations dictionary."""
        ops = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b, eps=1e-12: a / (b + eps),
            "pow": lambda a, p: a ** p,
            "gt":  lambda a, t: (a > t).astype(float),
            "lt":  lambda a, t: (a < t).astype(float),
            "clip": lambda a, lo, hi: a.clip(lo, hi),
            "zscore": self._op_zscore,
            "rolling_mean": self._op_rolling_mean,
            "rolling_std": self._op_rolling_std,
            "log": self._op_log
        }
        return ops

    def _op_zscore(self, s: pd.Series) -> pd.Series:
        """Z-score operation with VectorBT optimization."""
        if VECTORBT_AVAILABLE and zscore is not None:
            return zscore(s)
        m, sd = s.mean(), s.std(ddof=0) + 1e-12
        return (s - m) / sd

    def _op_rolling_mean(self, s: pd.Series, window: int) -> pd.Series:
        """Rolling mean operation with VectorBT optimization."""
        if VECTORBT_AVAILABLE and rolling_mean is not None:
            return rolling_mean(s, window=window)
        return s.rolling(window).mean()

    def _op_rolling_std(self, s: pd.Series, window: int) -> pd.Series:
        """Rolling std operation with VectorBT optimization."""
        if VECTORBT_AVAILABLE and rolling_std is not None:
            return rolling_std(s, window=window)
        return s.rolling(window).std(ddof=0)

    def _op_log(self, s: pd.Series) -> pd.Series:
        """Log operation with safe handling."""
        return np.log(np.abs(s) + 1e-12)

    def generate_interactions(self,
                            materialized_htfs: Dict[str, Any],
                            base_features: Union[pd.DataFrame, Dict[str, pd.Series], None],
                            targets: Optional[pd.Series] = None) -> List[GeneratedInteraction]:
        """
        Generate interactions from templates using VectorBT optimization.

        Args:
            materialized_htfs: Materialized HTF features
            base_features: Base features
            targets: Target variables

        Returns:
            List of generated interactions
        """
        start_time = time.time()

        def _generate_interactions():
            tprint_info("🎯 Starting template-based interaction generation...")

            # Build feature stores
            base_store = FeatureStore(base_features) if base_features is not None else None
            htf_store = FeatureStore({name: getattr(obj, "series", getattr(obj, "values", None))
                                    for name, obj in materialized_htfs.items()
                                    if hasattr(obj, "series") or hasattr(obj, "values")}) if materialized_htfs else None

            tprint_debug(f"📊 Base features: {len(base_store.names()) if base_store else 0}")
            tprint_debug(f"📊 HTF features: {len(htf_store.names()) if htf_store else 0}")

            # Determine budget allocation
            budget_allocation = self._determine_budget_allocation(materialized_htfs)
            tprint_debug(f"📊 Budget allocation: {budget_allocation}")

            # Generate core interactions
            core_interactions = []
            if self.config.enable_core_templates:
                tprint_debug("Generating core interactions...")
                if self.config.enable_parallel and VECTORBT_AVAILABLE:
                    core_interactions = self._generate_core_interactions_parallel(
                        base_store, targets, budget_allocation['core']
                    )
                else:
                    core_interactions = self._generate_core_interactions_vectorbt(
                        base_store, targets, budget_allocation['core']
                    )
                tprint_success(f"✅ Generated {len(core_interactions)} core interactions")

            # Generate HTF-aware interactions
            htf_interactions = []
            if self.config.enable_htf_templates:
                tprint_debug("Generating HTF-aware interactions...")
                htf_interactions = self._generate_htf_interactions_vectorbt(
                    base_store, htf_store, targets, budget_allocation['htf_aware']
                )
                tprint_success(f"✅ Generated {len(htf_interactions)} HTF interactions")

            # Combine all interactions
            all_interactions = core_interactions + htf_interactions

            # Apply interaction heredity if enabled
            if self.config.enable_interaction_heredity:
                all_interactions = self._apply_interaction_heredity(all_interactions, base_store, htf_store)

            # Apply VectorBT-based feature selection
            selected_interactions = self._apply_vectorbt_feature_selection(all_interactions, targets)
            
            # Ensure all interactions are properly aligned
            selected_interactions = self._align_interactions(selected_interactions)

            # Update performance stats
            generation_time = time.time() - start_time
            self.performance_stats.update({
                'total_generations': 1,
                'successful_generations': 1,
                'total_generation_time': generation_time,
                'interactions_generated': len(selected_interactions)
            })

            tprint_success(f"✅ Template interaction generation completed in {generation_time:.3f}s")
            tprint_info(f"📊 Total interactions: {len(selected_interactions)}")
            tprint_info(f"📊 Core: {len(core_interactions)}, HTF: {len(htf_interactions)}")

            return selected_interactions

        # Execute with error handling
        try:
            return _generate_interactions()
        except Exception as e:
            tprint_error(f"❌ Template interaction generation failed: {e}")
            self.performance_stats['failed_generations'] += 1
            return []

    def _normalize_base_features(self, base_features: Union[pd.DataFrame, Dict[str, pd.Series], None]) -> Dict[str, pd.Series]:
        """Convert supported base feature structures into a column-keyed mapping."""
        normalized = {}

        if base_features is None:
            return normalized

        if isinstance(base_features, pd.DataFrame):
            for column in base_features.columns:
                series = base_features[column]
                if isinstance(series, pd.Series):
                    normalized[column] = series
        elif isinstance(base_features, dict):
            for name, series in base_features.items():
                if isinstance(series, pd.Series):
                    normalized[name] = series

        return normalized

    def _determine_budget_allocation(self, materialized_htfs: Dict[str, Any]) -> Dict[str, int]:
        """Determine budget allocation for different interaction types."""
        # Base budget
        total_budget = self.config.total_budget

        # Calculate HTF performance
        htf_utilities = []
        for feature_name, feature in materialized_htfs.items():
            if hasattr(feature, 'utility_score'):
                htf_utilities.append(feature.utility_score)

        avg_htf_utility = np.mean(htf_utilities) if htf_utilities else 0.0

        # Allocate budget based on HTF performance
        if avg_htf_utility > 0.1:  # Top-quartile performance
            core_budget = self.config.core_budget
            htf_aware_budget = self.config.htf_aware_budget
        else:
            # Standard allocation
            core_budget = min(self.config.core_budget + 5, total_budget)
            htf_aware_budget = min(self.config.htf_aware_budget - 5, total_budget - core_budget)

        return {
            'core': core_budget,
            'htf_aware': htf_aware_budget
        }

    def _generate_core_interactions_vectorbt(self,
                                           base_store: FeatureStore,
                                           targets: Optional[pd.Series],
                                           budget: int) -> List[GeneratedInteraction]:
        """Generate core interactions using VectorBT optimization."""
        interactions = []

        if not VECTORBT_AVAILABLE:
            tprint_warning("VectorBT not available, using fallback method")
            return self._generate_core_interactions_fallback(base_store, targets, budget)

        try:
            # Group features by type using store
            feature_groups = self._group_features_by_type_from_store(base_store)

            # Generate interactions for each template
            for template in self.core_templates.templates[:budget]:
                template_interactions = self._generate_template_interactions_vectorbt(
                    template, feature_groups, targets, base_store=base_store
                )
                interactions.extend(template_interactions)

        except Exception as e:
            tprint_error(f"VectorBT core interactions failed: {e}, using fallback")
            return self._generate_core_interactions_fallback(base_store, targets, budget)

        return interactions

    def _generate_core_interactions_parallel(self,
                                           base_store: FeatureStore,
                                           targets: Optional[pd.Series],
                                           budget: int) -> List[GeneratedInteraction]:
        """Generate core interactions with parallel processing."""
        interactions = []
        
        if not VECTORBT_AVAILABLE:
            return self._generate_core_interactions_fallback(base_store, targets, budget)

        try:
            # Group features by type using store
            feature_groups = self._group_features_by_type_from_store(base_store)

            # Process templates in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_template = {
                    executor.submit(
                        self._generate_template_interactions_vectorbt,
                        template, feature_groups, targets, base_store
                    ): template for template in self.core_templates.templates[:budget]
                }

                for future in as_completed(future_to_template):
                    try:
                        template_interactions = future.result()
                        interactions.extend(template_interactions)
                    except Exception as e:
                        template = future_to_template[future]
                        tprint_warning(f"Template {template.name} failed in parallel: {e}")

        except Exception as e:
            tprint_error(f"Parallel core interactions failed: {e}, using fallback")
            return self._generate_core_interactions_fallback(base_store, targets, budget)

        return interactions

    def _generate_htf_interactions_vectorbt(self,
                                          base_store: FeatureStore,
                                          htf_store: FeatureStore,
                                          targets: Optional[pd.Series],
                                          budget: int) -> List[GeneratedInteraction]:
        """Generate HTF-aware interactions using VectorBT optimization."""
        interactions = []

        if not VECTORBT_AVAILABLE:
            tprint_warning("VectorBT not available, using fallback method")
            return self._generate_htf_interactions_fallback(base_store, htf_store, targets, budget)

        try:
            # Group features by type using stores
            htf_groups = self._group_htf_features_by_type_from_store(htf_store)
            base_groups = self._group_features_by_type_from_store(base_store)

            # Generate interactions for each template
            for template in self.htf_aware_templates.templates[:budget]:
                template_interactions = self._generate_htf_template_interactions_vectorbt(
                    template, htf_groups, base_groups, targets, base_store, htf_store
                )
                interactions.extend(template_interactions)

        except Exception as e:
            tprint_error(f"VectorBT HTF interactions failed: {e}, using fallback")
            return self._generate_htf_interactions_fallback(base_store, htf_store, targets, budget)

        return interactions

    def _group_features_by_type(self, features: Dict[str, pd.Series]) -> Dict[str, List[str]]:
        """Group features by type."""
        groups = {
            'price_feature': [],
            'volatility_feature': [],
            'momentum_feature': [],
            'mean_reversion_feature': [],
            'liquidity_feature': [],
            'volume_feature': [],
            'tod_indicator': [],
            'regime_indicator': []
        }

        for name, series in features.items():
            name_lower = name.lower()
            if any(x in name_lower for x in ['price', 'close', 'open', 'high', 'low']):
                groups['price_feature'].append(name)
            elif any(x in name_lower for x in ['vol', 'sigma', 'rv', 'gk']):
                groups['volatility_feature'].append(name)
            elif any(x in name_lower for x in ['mom', 'momentum', 'signal', 'alpha']):
                groups['momentum_feature'].append(name)
            elif any(x in name_lower for x in ['rsi', 'stoch', 'mean_rev', 'osc']):
                groups['mean_reversion_feature'].append(name)
            elif any(x in name_lower for x in ['liquidity', 'depth', 'book']):
                groups['liquidity_feature'].append(name)
            elif 'volume' in name_lower:
                groups['volume_feature'].append(name)
            elif any(x in name_lower for x in ['tod', 'time_of_day', 'session']):
                groups['tod_indicator'].append(name)
            elif any(x in name_lower for x in ['regime', 'vol_regime']):
                groups['regime_indicator'].append(name)

        return groups

    def _group_features_by_type_from_store(self, store: FeatureStore) -> Dict[str, List[str]]:
        """Group features by type using FeatureStore."""
        groups = {
            'price_feature': [],
            'volatility_feature': [],
            'momentum_feature': [],
            'mean_reversion_feature': [],
            'liquidity_feature': [],
            'volume_feature': [],
            'tod_indicator': [],
            'regime_indicator': []
        }

        for role, names in store._registry_by_role.items():
            if role in groups:
                groups[role].extend(names)

        return groups

    def _group_htf_features_by_type(self, materialized_htfs: Dict[str, Any]) -> Dict[str, List[str]]:
        """Group HTF features by type."""
        groups = {
            'htf_trend_feature': [],
            'htf_volatility_feature': [],
            'htf_momentum_feature': [],
            'htf_anchor_feature': [],
            'htf_regime_feature': []
        }

        for name, feature in materialized_htfs.items():
            family = getattr(feature, 'family', 'unknown') or 'unknown'
            name_lower = name.lower()

            if any(x in name_lower for x in ['trend', 'ema', 'sma']):
                groups['htf_trend_feature'].append(name)
            elif any(x in name_lower for x in ['vol', 'sigma', 'rv', 'var']):
                groups['htf_volatility_feature'].append(name)
            elif any(x in name_lower for x in ['rsi', 'stoch', 'momentum', 'osc']):
                groups['htf_momentum_feature'].append(name)
            elif any(x in name_lower for x in ['vwap', 'anchor']):
                groups['htf_anchor_feature'].append(name)
            elif any(x in name_lower for x in ['regime', 'state']):
                groups['htf_regime_feature'].append(name)

        return groups

    def _group_htf_features_by_type_from_store(self, store: FeatureStore) -> Dict[str, List[str]]:
        """Group HTF features by type using FeatureStore."""
        groups = {
            'htf_trend_feature': [],
            'htf_volatility_feature': [],
            'htf_momentum_feature': [],
            'htf_anchor_feature': [],
            'htf_regime_feature': []
        }

        for role, names in store._registry_by_role.items():
            if role in groups:
                groups[role].extend(names)

        return groups

    def _generate_template_interactions_vectorbt(self,
                                               template: InteractionTemplate,
                                               feature_groups: Dict[str, List[str]],
                                               targets: Optional[pd.Series],
                                               base_store: FeatureStore = None) -> List[GeneratedInteraction]:
        """Generate interactions from a template using VectorBT optimization."""
        interactions = []

        try:
            # Get feature combinations for this template
            feature_combinations = self._get_feature_combinations(template, feature_groups, base_store)

            for combination in feature_combinations:
                try:
                    # Generate interaction using VectorBT
                    interaction_series = self._calculate_interaction_vectorbt(template, combination, base_store)

                    if interaction_series is not None and self._is_valid_interaction(interaction_series):
                        # Calculate utility score
                        utility_score = self._calculate_utility_score(interaction_series, targets)

                        if utility_score >= self.config.min_utility_score:
                            # Create interaction object
                            interaction = GeneratedInteraction(
                                name=f"{template.name}_{combination['name']}",
                                formula=template.formula,
                                parent_features=combination['features'],
                                interaction_type=template.template_type,
                                feature_series=interaction_series,
                                utility_score=utility_score,
                                metadata={
                                    'template': template.name,
                                    'combination': combination,
                                    'vectorbt_optimized': VECTORBT_AVAILABLE
                                }
                            )

                            interactions.append(interaction)

                except Exception as e:
                    tprint_debug(f"Template interaction generation failed: {e}")
                    continue

        except Exception as e:
            tprint_warning(f"Template {template.name} failed: {e}")

        return interactions

    def _get_feature_combinations(self,
                                 template: InteractionTemplate,
                                 feature_groups: Dict[str, List[str]],
                                 base_store: FeatureStore = None) -> List[Dict[str, Any]]:
        """Get feature combinations for a template."""
        combinations = []

        # Get required feature lists
        required_lists = [feature_groups.get(req, []) for req in template.required_features]

        # Generate Cartesian product
        for combo in product(*required_lists):
            combination = dict(zip(template.required_features, combo))
            combinations.append({
                'name': '_'.join(combo),
                'features': list(combo),
                'combination': combination,
                'params': template.metadata.get("defaults", {})
            })

        return combinations[:template.max_instances]

    def _calculate_interaction_vectorbt(self,
                                      template: InteractionTemplate,
                                      combination: Dict[str, Any],
                                      base_store: FeatureStore = None,
                                      htf_store: FeatureStore = None,
                                      params: Optional[Dict[str, Any]] = None) -> Optional[pd.Series]:
        """
        Calculate interaction using VectorBT optimization.
        
        Args:
            template: Interaction template
            combination: Feature combination dictionary
            base_store: Base feature store
            htf_store: HTF feature store
            params: Template parameters
        """
        params = params or {}
        try:
            # Build local resolver for required vars
            def S(name):
                # check both stores
                if base_store and name in base_store.names():
                    return base_store.get(name)
                if htf_store and name in htf_store.names():
                    return htf_store.get(name)
                # allow direct Series injection via combo
                return combination["combination"].get(name, None)

            # Pull Series for required roles
            role_to_name = combination["combination"]  # e.g. {"price_feature": "close", "volatility_feature": "rv"}
            series = {}
            for role, name in role_to_name.items():
                if base_store and name in base_store.names():
                    series[role] = base_store.get(name)
                elif htf_store and name in htf_store.names():
                    series[role] = htf_store.get(name)
                else:
                    tprint_debug(f"Feature {name} not found in stores")
                    return None

            # Dispatch by template
            name = template.name

            if name == "price_vol_interaction":
                s = self.formula_ops["mul"](series["price_feature"], series["volatility_feature"])

            elif name == "momentum_meanrev_interaction":
                s = self.formula_ops["mul"](series["momentum_feature"], series["mean_reversion_feature"])

            elif name == "liquidity_price_interaction":
                s = self.formula_ops["mul"](series["liquidity_feature"], series["price_feature"])

            elif name == "vol_volume_interaction":
                s = self.formula_ops["mul"](series["volatility_feature"], series["volume_feature"])

            elif name == "tod_interaction":
                s = self.formula_ops["mul"](series["feature"], series["tod_indicator"])

            elif name == "cross_sectional_interaction":
                s = self.formula_ops["sub"](series["feature"], series["market_feature"])

            elif name == "regime_interaction":
                s = self.formula_ops["mul"](series["feature"], series["regime_indicator"])

            elif name == "lag_interaction":
                s = self.formula_ops["mul"](series["feature"], series["feature_lag"])

            elif name == "polynomial_interaction":
                s = self.formula_ops["pow"](series["feature"], 2)

            elif name == "ratio_interaction":
                eps = params.get("epsilon", 1e-6)
                s = self.formula_ops["div"](series["feature1"], series["feature2"], eps)

            elif name == "difference_interaction":
                s = self.formula_ops["sub"](series["feature1"], series["feature2"])

            elif name == "product_interaction":
                s = self.formula_ops["mul"](series["feature1"], series["feature2"])

            elif name == "conditional_interaction":
                th = params.get("threshold", 0.0)
                mask = self.formula_ops["gt"](series["condition"], th)
                s = self.formula_ops["mul"](series["feature"], mask)

            elif name == "rolling_interaction":
                window = int(params.get("window", 20))
                s = self._op_rolling_mean(series["feature"], window)

            elif name == "zscore_interaction":
                s = self._op_zscore(series["feature"])

            # HTF-aware
            elif name == "htf_trend_liquidity_interaction":
                s = self.formula_ops["mul"](series["htf_trend_feature"], series["base_liquidity_feature"])

            elif name == "htf_vol_signal_interaction":
                s = self.formula_ops["mul"](series["htf_volatility_feature"], series["base_signal_feature"])

            elif name == "htf_momentum_conflict_interaction":
                s = self.formula_ops["mul"](series["htf_momentum_feature"], -series["base_momentum_feature"])

            elif name == "htf_regime_base_interaction":
                s = self.formula_ops["mul"](series["htf_regime_feature"], series["base_feature"])

            elif name == "htf_anchor_deviation_interaction":
                s = self.formula_ops["mul"](series["htf_anchor_feature"], series["base_deviation_feature"])

            else:
                tprint_debug(f"Unknown template: {name}")
                return None

            # Ensure proper naming and alignment
            s.name = f"{template.name}_{combination['name']}"
            return s

        except Exception as e:
            tprint_debug(f"VectorBT interaction calc error [{template.name}]: {e}")
            return None

    def _generate_htf_template_interactions_vectorbt(self,
                                                   template: InteractionTemplate,
                                                   htf_groups: Dict[str, List[str]],
                                                   base_groups: Dict[str, List[str]],
                                                   targets: Optional[pd.Series],
                                                   base_store: FeatureStore = None,
                                                   htf_store: FeatureStore = None) -> List[GeneratedInteraction]:
        """Generate HTF template interactions using VectorBT optimization."""
        interactions = []

        try:
            # Get feature combinations for this template
            feature_combinations = self._get_htf_feature_combinations(template, htf_groups, base_groups, base_store, htf_store)

            for combination in feature_combinations:
                try:
                    # Generate interaction using VectorBT
                    interaction_series = self._calculate_interaction_vectorbt(template, combination, base_store, htf_store, combination.get("params"))

                    if interaction_series is not None and self._is_valid_interaction(interaction_series):
                        # Calculate utility score
                        utility_score = self._calculate_utility_score(interaction_series, targets)

                        if utility_score >= self.config.min_utility_score:
                            # Create interaction object
                            interaction = GeneratedInteraction(
                                name=f"{template.name}_{combination['name']}",
                                formula=template.formula,
                                parent_features=combination['features'],
                                interaction_type=template.template_type,
                                feature_series=interaction_series,
                                utility_score=utility_score,
                                metadata={
                                    'template': template.name,
                                    'combination': combination,
                                    'vectorbt_optimized': VECTORBT_AVAILABLE
                                }
                            )

                            interactions.append(interaction)

                except Exception as e:
                    tprint_debug(f"HTF template interaction generation failed: {e}")
                    continue

        except Exception as e:
            tprint_warning(f"HTF template {template.name} failed: {e}")

        return interactions

    def _get_htf_feature_combinations(self,
                                    template: InteractionTemplate,
                                    htf_groups: Dict[str, List[str]],
                                    base_groups: Dict[str, List[str]],
                                    base_store: FeatureStore = None,
                                    htf_store: FeatureStore = None) -> List[Dict[str, Any]]:
        """Get HTF feature combinations for a template."""
        combinations = []

        # Determine pool per required role
        def pool_for_role(role):
            names = []
            if base_store:
                names.extend([n for n in base_store.by_role(role)])
            if htf_store:
                names.extend([n for n in htf_store.by_role(role)])
            # fall back to guessed groups for backward-compat
            names.extend(htf_groups.get(role, []))
            names.extend(base_groups.get(role, []))
            return list(dict.fromkeys(names))  # unique

        required_lists = [pool_for_role(req) for req in template.required_features]
        for tup in product(*required_lists):
            mapping = dict(zip(template.required_features, tup))
            combinations.append({
                "name": '_'.join(tup),
                "features": list(tup),
                "combination": mapping,
                "params": template.metadata.get("defaults", {})
            })

        return combinations[:template.max_instances]

    def _is_valid_interaction(self, series: pd.Series) -> bool:
        """Check if an interaction series is valid."""
        if series is None or series.empty:
            return False

        # Check for all NaN values
        if series.isna().all():
            return False

        # Check for infinite values
        if np.isinf(series).any():
            return False

        # Check for constant values (no variance)
        if series.nunique() <= 1:
            return False

        return True

    def _calculate_utility_score(self, interaction_series: pd.Series, targets: Optional[pd.Series]) -> float:
        """Calculate robust utility score for an interaction."""
        try:
            s = interaction_series.dropna()
            if s.empty:
                return 0.0
                
            if targets is None:
                # Unsupervised proxy
                var_score = float(s.var())
                # Anti-noise regularizer
                nz = s.ne(0).mean()
                return max(0.0, var_score) * (0.5 + 0.5 * nz)

            aligned = pd.concat([s, targets], axis=1).dropna()
            if len(aligned) < 50:
                return 0.0
                
            x, y = aligned.iloc[:,0], aligned.iloc[:,1]
            pear = x.corr(y) or 0.0
            spear = x.corr(y, method="spearman") or 0.0
            
            # Rolling IC stability
            roll_ic = x.rolling(100).corr(y).dropna()
            stab = 1.0 - roll_ic.std() if len(roll_ic) >= 10 else 0.0
            
            # Blend scores
            return float(0.5*abs(pear) + 0.4*abs(spear) + 0.1*max(0.0, stab))
            
        except Exception as e:
            tprint_debug(f"Utility score calculation failed: {e}")
            return 0.0

    def _apply_interaction_heredity(self, interactions: List[GeneratedInteraction], 
                                  base_store: FeatureStore = None, 
                                  htf_store: FeatureStore = None) -> List[GeneratedInteraction]:
        """Apply interaction heredity (keep ≥1 parent if interaction survives)."""
        if not interactions:
            return interactions
            
        keep = {i.name: i for i in interactions}
        parent_needed = set()

        for i in interactions:
            for p in i.parent_features:
                parent_needed.add(p)

        # Create lightweight GeneratedInteraction wrappers for missing parents
        for p in parent_needed:
            if p in keep:
                continue
                
            # Find parent series from stores
            parent_series = None
            if base_store and p in base_store.names():
                parent_series = base_store.get(p)
            elif htf_store and p in htf_store.names():
                parent_series = htf_store.get(p)
                
            if parent_series is not None:
                # Synthesize a parent feature as passthrough
                gi = GeneratedInteraction(
                    name=p,
                    formula="identity",
                    parent_features=[p],
                    interaction_type="parent",
                    feature_series=parent_series,
                    utility_score=0.0,
                    metadata={"heredity": True}
                )
                keep[p] = gi

        return list(keep.values())

    def _apply_vectorbt_feature_selection(self,
                                        interactions: List[GeneratedInteraction],
                                        targets: Optional[pd.Series]) -> List[GeneratedInteraction]:
        """Apply VectorBT-based feature selection."""
        if not interactions:
            return interactions

        try:
            # Sort by utility score
            interactions.sort(key=lambda x: x.utility_score, reverse=True)

            # Select top interactions
            max_interactions = min(len(interactions), self.config.total_budget)
            selected = interactions[:max_interactions]

            # Apply additional VectorBT-based filtering
            if VECTORBT_AVAILABLE and targets is not None:
                selected = self._filter_correlated_interactions(selected, targets)

            return selected

        except Exception as e:
            tprint_warning(f"VectorBT feature selection failed: {e}, returning all interactions")
            return interactions

    def _filter_correlated_interactions(self,
                                      interactions: List[GeneratedInteraction],
                                      targets: Optional[pd.Series]) -> List[GeneratedInteraction]:
        """Filter highly correlated interactions using greedy forward selection."""
        if len(interactions) <= 1:
            return interactions

        try:
            # Greedy selection by descending utility with redundancy check
            chosen = []
            chosen_df = None
            max_r = self.config.max_correlation_threshold

            for cand in sorted(interactions, key=lambda x: x.utility_score, reverse=True):
                s = cand.feature_series
                if s is None or s.isna().all():
                    continue
                    
                if chosen_df is None:
                    chosen.append(cand)
                    chosen_df = pd.DataFrame({cand.name: s})
                    continue
                    
                df = pd.concat([chosen_df, s.rename(cand.name)], axis=1).dropna()
                if df.shape[0] < 50:
                    continue
                    
                # Check max absolute corr vs existing chosen
                c = df.corr().iloc[:-1, -1].abs().max()
                if c <= max_r:
                    chosen.append(cand)
                    chosen_df = df.dropna()

            return chosen

        except Exception as e:
            tprint_warning(f"Correlation filtering failed: {e}")
            return interactions

    # Fallback methods for when VectorBT is not available
    def _generate_core_interactions_fallback(self,
                                           base_store: FeatureStore,
                                           targets: Optional[pd.Series],
                                           budget: int) -> List[GeneratedInteraction]:
        """Fallback method for core interactions when VectorBT is not available."""
        # Simplified fallback implementation
        interactions = []

        # Generate basic product interactions
        feature_names = list(base_store.names())
        for i, feat1 in enumerate(feature_names):
            for feat2 in feature_names[i+1:]:
                if len(interactions) >= budget:
                    break

                try:
                    s1 = base_store.get(feat1)
                    s2 = base_store.get(feat2)
                    product = s1 * s2
                    utility_score = self._calculate_utility_score(product, targets)

                    if utility_score >= self.config.min_utility_score:
                        interaction = GeneratedInteraction(
                            name=f"product_{feat1}_{feat2}",
                            formula=f"{feat1} * {feat2}",
                            parent_features=[feat1, feat2],
                            interaction_type="core",
                            feature_series=product,
                            utility_score=utility_score,
                            metadata={'fallback': True}
                        )
                        interactions.append(interaction)

                except Exception as e:
                    continue

        return interactions

    def _generate_htf_interactions_fallback(self,
                                          base_store: FeatureStore,
                                          htf_store: FeatureStore,
                                          targets: Optional[pd.Series],
                                          budget: int) -> List[GeneratedInteraction]:
        """Fallback method for HTF interactions when VectorBT is not available."""
        # Simplified fallback implementation
        interactions = []

        # Generate basic HTF interactions
        htf_names = list(htf_store.names()) if htf_store else []
        base_names = list(base_store.names()) if base_store else []

        for htf_name in htf_names:
            for base_name in base_names:
                if len(interactions) >= budget:
                    break

                try:
                    htf_series = htf_store.get(htf_name)
                    base_series = base_store.get(base_name)
                    product = htf_series * base_series
                    utility_score = self._calculate_utility_score(product, targets)

                    if utility_score >= self.config.min_utility_score:
                        interaction = GeneratedInteraction(
                            name=f"htf_{htf_name}_{base_name}",
                            formula=f"htf_{htf_name} * {base_name}",
                            parent_features=[htf_name, base_name],
                            interaction_type="htf_aware",
                            feature_series=product,
                            utility_score=utility_score,
                            metadata={'fallback': True}
                        )
                        interactions.append(interaction)

                except Exception as e:
                    continue

        return interactions

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()

    def to_dataframe(self, interactions: List[GeneratedInteraction]) -> pd.DataFrame:
        """Convert interactions to aligned DataFrame."""
        if not interactions:
            return pd.DataFrame()
        
        try:
            # Create DataFrame from interactions
            data = {}
            for interaction in interactions:
                if interaction.feature_series is not None:
                    data[interaction.name] = interaction.feature_series
            
            if not data:
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            return df.sort_index()
            
        except Exception as e:
            tprint_warning(f"Failed to create DataFrame from interactions: {e}")
            return pd.DataFrame()

    def _align_interactions(self, interactions: List[GeneratedInteraction]) -> List[GeneratedInteraction]:
        """Ensure all interactions are properly aligned to a common index."""
        if not interactions:
            return interactions
            
        try:
            # Find common index
            indices = [i.feature_series.index for i in interactions if i.feature_series is not None]
            if not indices:
                return interactions
                
            common_index = indices[0]
            for idx in indices[1:]:
                common_index = common_index.intersection(idx)
                
            if len(common_index) == 0:
                tprint_warning("No common index found for interactions")
                return interactions
                
            # Align all interactions
            aligned_interactions = []
            for interaction in interactions:
                if interaction.feature_series is not None:
                    aligned_series = interaction.feature_series.reindex(common_index)
                    aligned_series = aligned_series.astype("float64")
                    
                    # Create new interaction with aligned series
                    aligned_interaction = GeneratedInteraction(
                        name=interaction.name,
                        formula=interaction.formula,
                        parent_features=interaction.parent_features,
                        interaction_type=interaction.interaction_type,
                        feature_series=aligned_series,
                        utility_score=interaction.utility_score,
                        metadata=interaction.metadata
                    )
                    aligned_interactions.append(aligned_interaction)
                else:
                    aligned_interactions.append(interaction)
                    
            return aligned_interactions
            
        except Exception as e:
            tprint_warning(f"Failed to align interactions: {e}")
            return interactions

# Convenience functions
def create_template_interaction_generator(config: Optional[TemplateConfig] = None) -> TemplateInteractionGenerator:
    """Create a template interaction generator with default configuration."""
    return TemplateInteractionGenerator(config)

def generate_template_interactions(materialized_htfs: Dict[str, Any],
                                 base_features: Union[pd.DataFrame, Dict[str, pd.Series], None],
                                 targets: Optional[pd.Series] = None,
                                 config: Optional[TemplateConfig] = None) -> List[GeneratedInteraction]:
    """
    Convenience function to generate template-based interactions.

    Args:
        materialized_htfs: Materialized HTF features
        base_features: Base features
        targets: Target variables
        config: Optional configuration

    Returns:
        List of generated interactions
    """
    generator = create_template_interaction_generator(config)
    return generator.generate_interactions(materialized_htfs, base_features, targets)

# Export main classes and functions
__all__ = [
    'TemplateInteractionGenerator',
    'CoreInteractionTemplates',
    'HTFAwareTemplates',
    'InteractionTemplate',
    'GeneratedInteraction',
    'TemplateConfig',
    'create_template_interaction_generator',
    'generate_template_interactions'
]
