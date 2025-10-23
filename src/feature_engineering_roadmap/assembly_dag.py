"""
Assembly DAG for End-to-End Roadmap

Orchestrates the complete feature engineering pipeline:
- Calendar sessionization
- Parent feature computation
- Transform application
- Interaction creation
- Patch/GRU model integration
- Feature selection and assembly

VectorBT Optimizations:
- Vectorized feature assembly operations
-
- Parallel processing for feature selection
- Memory-efficient operations
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import warnings

import pandas as pd
import numpy as np

# VectorBT imports for optimization
try:
    import vectorbt as vbt
    from vectorbt.utils.array_ops import rolling_apply
    from vectorbt.utils.array_ops import rolling_apply_parallel
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_apply = None
    rolling_apply_parallel = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# GPU acceleration removed - CuPy not supported on all platforms
CUPY_AVAILABLE = False

# Import our modules
from .data_contracts import InputBar, FeatureStore, ArtifactsRegistry
from .feature_registry import FeatureRegistry, PriceReturnsFeatures, VolatilityFeatures, MeanReversionFeatures, LiquidityMicroFeatures, AnchorsTODFeatures, ContextFeatures
from .transforms import TransformRouter, create_default_transform_config, apply_winsorization
from .lookback_selection import LookbackSelector, create_feature_families
from .interactions import InteractionEngine, create_default_interaction_config
from ..models.patch_gru import PatchOrchestrator, PatchConfig, ModelType

class AssemblyStatus(Enum):
    """Status of assembly process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AssemblyConfig:
    """Configuration for assembly process with VectorBT optimization."""
    feature_budget_pre: int = 120
    feature_budget_post: Tuple[int, int] = (30, 60)
    interactions_cap: int = 15
    transforms_per_parent: int = 1
    lookback_ceiling_minutes: int = 120
    latency_budget_ms: int = 50
    patch_model_type: ModelType = ModelType.GRU
    patch_sequence_length: int = 24  # 2h at 5min bars
    patch_horizons: List[int] = None

    # VectorBT optimization settings
    use_vectorbt: bool = True
    use_gpu: bool = False
    enable_parallel: bool = True
    performance_threshold: int = 1000  # Minimum samples for VectorBT optimization

    def __post_init__(self):
        if self.patch_horizons is None:
            self.patch_horizons = [1, 3]

@dataclass
class AssemblyResult:
    """Result of assembly process."""
    features: pd.DataFrame
    feature_names: List[str]
    selected_features: List[str]
    patch_features: Dict[str, pd.Series]
    artifacts: ArtifactsRegistry
    status: AssemblyStatus
    metadata: Dict[str, Any]

class CalendarSessionizer:
    """Exchange-aware session management."""

    def __init__(self, exchange: str = "NYSE"):
        self.exchange = exchange
        self.sessions = {}

    def sessionize(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add session information to bars."""
        result = bars.copy()

        # Simple sessionization (would be more complex in practice)
        if 'timestamp' in result.columns:
            timestamp_series = pd.to_datetime(result['timestamp'])
        else:
            timestamp_series = pd.to_datetime(result.index)

        timestamp_series = pd.Series(timestamp_series.values, index=result.index)
        result['session_id'] = timestamp_series.dt.tz_localize(None).dt.date

        session_starts = timestamp_series.groupby(result['session_id']).transform('min')
        session_ends = timestamp_series.groupby(result['session_id']).transform('max')

        minutes_from_open = (timestamp_series - session_starts).dt.total_seconds() / 60.0
        minutes_to_close = (session_ends - timestamp_series).dt.total_seconds() / 60.0

        result['open30'] = (minutes_from_open <= 30).astype(int)
        result['last30'] = (minutes_to_close <= 30).astype(int)

        return result

class ParentFeatureBuilder:
    """Builds parent features from market data."""

    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self.feature_builders = {
            'price_returns': PriceReturnsFeatures,
            'volatility': VolatilityFeatures,
            'mean_reversion': MeanReversionFeatures,
            'liquidity_micro': LiquidityMicroFeatures,
            'anchors_tod': AnchorsTODFeatures,
            'context': ContextFeatures
        }

    def build_all_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Build all parent features."""
        features = {}

        # Price/Returns features
        features['p/r1'] = PriceReturnsFeatures.r1(bars)
        features['p/r3'] = PriceReturnsFeatures.r3(bars)
        features['p/r5'] = PriceReturnsFeatures.r5(bars)
        features['p/r10'] = PriceReturnsFeatures.r10(bars)
        features['p/mom5'] = PriceReturnsFeatures.mom5(bars)
        features['p/mom10'] = PriceReturnsFeatures.mom10(bars)
        features['p/mom20'] = PriceReturnsFeatures.mom20(bars)
        features['p/price_ema10_pct'] = PriceReturnsFeatures.price_ema10_pct(bars)
        features['p/price_ema20_pct'] = PriceReturnsFeatures.price_ema20_pct(bars)
        features['p/bollz20'] = PriceReturnsFeatures.bollz20(bars)

        # Volatility features
        features['p/sigma_ew'] = VolatilityFeatures.sigma_ew(bars, halflife=12)
        features['p/gk_w'] = VolatilityFeatures.gk_w(bars, window=12)
        features['p/rv_bipower_12'] = VolatilityFeatures.rv_bipower_12(bars)
        features['p/rv_short_3'] = VolatilityFeatures.rv_short_3(bars)
        features['p/sigma_slope_6'] = VolatilityFeatures.sigma_slope_6(bars)
        features['p/range_pct'] = VolatilityFeatures.range_pct(bars)

        # Mean reversion features
        features['p/rsi7'] = MeanReversionFeatures.rsi7(bars)
        features['p/rsi14'] = MeanReversionFeatures.rsi14(bars)
        features['p/stochk14'] = MeanReversionFeatures.stochk14(bars)
        features['p/autocorr_r1_w'] = MeanReversionFeatures.autocorr_r1_w(bars, window=12)

        # Liquidity/Micro features (book-optional)
        features['p/volume_z18'] = LiquidityMicroFeatures.volume_z18(bars)
        features['p/tradecount_z18'] = LiquidityMicroFeatures.tradecount_z18(bars)
        features['p/spread_z18'] = LiquidityMicroFeatures.spread_z18(bars)
        features['p/dollarvol_z18'] = LiquidityMicroFeatures.dollarvol_z18(bars)
        features['p/ofi_proxy'] = LiquidityMicroFeatures.ofi_proxy(bars)
        features['p/microprice_dev'] = LiquidityMicroFeatures.microprice_dev(bars)

        # Anchors & TOD features
        features['p/vwap_session_dist'] = AnchorsTODFeatures.vwap_session_dist(bars)
        features['p/vwap_roll12_dist'] = AnchorsTODFeatures.vwap_roll12_dist(bars)
        features['p/open30'] = AnchorsTODFeatures.open30(bars)
        features['p/last30'] = AnchorsTODFeatures.last30(bars)

        # Context features (optional)
        features['p/beta30'] = ContextFeatures.beta30(bars)
        features['p/mkt_dispersion'] = ContextFeatures.mkt_dispersion(bars)

        # Remove features with all NaN values
        features_df = pd.DataFrame(features, index=bars.index)
        features_df = features_df.dropna(axis=1, how='all')

        return features_df

class AssemblyError(RuntimeError):
    """Raised when assembly fails."""

class AssemblyDAG:
    """Main assembly DAG orchestrator with VectorBT optimization."""

    def __init__(self, config: AssemblyConfig):
        self.config = config
        self.registry = FeatureRegistry()
        self.sessionizer = CalendarSessionizer()
        self.feature_builder = ParentFeatureBuilder(self.registry)
        self.status = AssemblyStatus.PENDING
        self.artifacts = None
        self.logger = logging.getLogger(__name__)

        # VectorBT optimization settings
        self.use_vectorbt = config.use_vectorbt and VECTORBT_AVAILABLE
        self.use_gpu = False  # GPU support removed
        self.enable_parallel = config.enable_parallel and VECTORBT_AVAILABLE

    def assemble(self,
                 bars: pd.DataFrame,
                 targets: Optional[Dict[int, pd.Series]] = None) -> AssemblyResult:
        """Assemble complete feature pipeline."""

        self.status = AssemblyStatus.IN_PROGRESS
        rotation_metadata: Dict[str, Any] = {}

        try:
            # Step 1: Sessionize bars
            bars_sessionized = self.sessionizer.sessionize(bars)

            # Step 2: Build parent features
            parent_features = self.feature_builder.build_all_features(bars_sessionized)

            # Step 3: Lookback selection
            lookback_selector = LookbackSelector()
            feature_families = create_feature_families(parent_features.columns.tolist())
            lookback_choices = lookback_selector.select_lookbacks(
                parent_features,
                targets.get(1, pd.Series(0, index=parent_features.index)) if targets else pd.Series(0, index=parent_features.index),
                feature_families
            )

            # Step 4: Transform features
            transform_config = create_default_transform_config(parent_features.columns.tolist())
            transform_router = TransformRouter(transform_config)

            # Split data for transform fitting
            split_idx = int(len(parent_features) * 0.8)
            train_features = parent_features.iloc[:split_idx]
            val_features = parent_features.iloc[split_idx:]

            transformed_results = transform_router.fit_transform(train_features, val_features)

            # Combine transformed features and restore original order
            combined_transformed = []
            for feature_name, results in transformed_results.items():
                train_df = results.get('train', pd.DataFrame(index=train_features.index))
                val_df = results.get('val', pd.DataFrame(index=val_features.index))
                feature_df = pd.concat([train_df, val_df], axis=0).sort_index()
                combined_transformed.append(feature_df)

            if combined_transformed:
                transformed_features = pd.concat(combined_transformed, axis=1)
                transformed_features = transformed_features.reindex(parent_features.index)
            else:
                transformed_features = pd.DataFrame(index=parent_features.index)

            # Apply winsorization
            transformed_features = apply_winsorization(transformed_features)

            if not transformed_features.empty:
                transformed_features, rotation_metadata = self._orthogonalize_correlated_features(
                    transformed_features
                )

            # Step 5: Patch/GRU model
            patch_features = {}
            if targets:
                patch_config = PatchConfig(
                    model_type=self.config.patch_model_type,
                    sequence_length=self.config.patch_sequence_length,
                    horizons=self.config.patch_horizons
                )
                patch_orchestrator = PatchOrchestrator(patch_config)

                # Get OOF predictions
                oof_predictions = patch_orchestrator.get_oof_predictions(
                    bars_sessionized, targets, n_folds=3
                )

                patch_features = {
                    'y_hat_h1': oof_predictions.y_hat_h1,
                    'y_hat_h3': oof_predictions.y_hat_h3,
                    'y_hat_conf': oof_predictions.y_hat_conf
                }

            # Step 6: Create interactions
            interaction_config = create_default_interaction_config()
            interaction_engine = InteractionEngine(interaction_config)

            interactions = interaction_engine.build_interactions(
                transformed_features, patch_features
            )

            # Step 7: Assemble final feature matrix
            final_features = pd.concat([transformed_features, interactions], axis=1)

            # Step 8: Feature selection (pre-filter to budget)
            selected_features = self._select_features(
                final_features,
                targets.get(1, pd.Series(0, index=final_features.index)) if targets else None
            )

            # Step 9: Create artifacts
            self.artifacts = self._create_artifacts(
                transform_router, lookback_choices, interaction_config, patch_features, rotation_metadata
            )

            self.status = AssemblyStatus.COMPLETED

            return AssemblyResult(
                features=final_features[selected_features] if selected_features else final_features,
                feature_names=selected_features if selected_features else final_features.columns.tolist(),
                selected_features=selected_features if selected_features else [],
                patch_features=patch_features,
                artifacts=self.artifacts,
                status=self.status,
                metadata={
                    'total_features': len(final_features.columns),
                    'selected_features': len(selected_features) if selected_features else 0,
                    'parent_features': len(parent_features.columns),
                    'transformed_features': len(transformed_features.columns),
                    'interactions': len(interactions.columns),
                    'patch_features': len(patch_features),
                    'orthogonalized_groups': len(rotation_metadata)
                }
            )

        except Exception as exc:
            self.status = AssemblyStatus.FAILED
            self.logger.exception("Assembly failed")
            raise AssemblyError("Feature assembly failed") from exc

    def _select_features(self,
                        features: pd.DataFrame,
                        targets: Optional[pd.Series]) -> List[str]:
        """Select features within budget constraints."""

        if targets is None or len(features.columns) <= self.config.feature_budget_pre:
            return features.columns.tolist()[:self.config.feature_budget_pre]

        # Simple correlation-based selection
        correlations = []
        for col in features.columns:
            if not features[col].isna().all() and not targets.isna().all():
                corr = features[col].corr(targets)
                if not pd.isna(corr):
                    correlations.append((col, abs(corr)))

        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)

        # Select top features within budget
        selected = [col for col, _ in correlations[:self.config.feature_budget_pre]]

        return selected

    def _orthogonalize_correlated_features(self,
                                           features: pd.DataFrame,
                                           threshold: float = 0.9
                                           ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Replace highly correlated feature groups with orthogonal rotations using VectorBT optimization."""

        if features.shape[1] < 2:
            return features, {}

        if self.use_vectorbt and len(features) > self.config.performance_threshold:
            return self._orthogonalize_correlated_features_vectorized(features, threshold)
        else:
            return self._orthogonalize_correlated_features_sequential(features, threshold)

    def _orthogonalize_correlated_features_sequential(self,
                                                    features: pd.DataFrame,
                                                    threshold: float = 0.9
                                                    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Sequential implementation for small datasets."""
        corr_matrix = features.corr().abs()
        corr_matrix = corr_matrix.fillna(0.0)
        np.fill_diagonal(corr_matrix.values, 0.0)

        visited = set()
        rotation_metadata: Dict[str, Any] = {}

        for column in corr_matrix.columns:
            if column in visited:
                continue

            group = {column}
            stack = [column]

            while stack:
                current = stack.pop()
                visited.add(current)
                strong_partners = corr_matrix.loc[current][corr_matrix.loc[current] > threshold].index.tolist()
                for partner in strong_partners:
                    if partner not in group:
                        group.add(partner)
                        if partner not in visited:
                            stack.append(partner)

            if len(group) < 2:
                continue

            ordered_group = sorted(group, key=lambda name: list(features.columns).index(name))
            subset = features[ordered_group]

            column_means = subset.mean(axis=0)
            subset_filled = subset.fillna(column_means)
            centered = subset_filled - column_means

            if np.allclose(centered.values, 0.0):
                continue

            try:
                _, _, vh = np.linalg.svd(centered.values, full_matrices=False)
            except np.linalg.LinAlgError:
                continue

            rotation_matrix = vh.T
            rotated_values = centered.values @ rotation_matrix

            rotated_df = pd.DataFrame(rotated_values, index=subset.index, columns=ordered_group)
            features.loc[:, ordered_group] = rotated_df

            group_key = "::".join(ordered_group)
            rotation_metadata[group_key] = {
                'columns': ordered_group,
                'means': column_means.to_dict(),
                'rotation_matrix': rotation_matrix.tolist(),
                'method': 'pca',
                'threshold': threshold
            }

        return features, rotation_metadata

    def _orthogonalize_correlated_features_vectorized(self,
                                                    features: pd.DataFrame,
                                                    threshold: float = 0.9
                                                    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """VectorBT-optimized correlation analysis and orthogonalization."""
        if False:  # GPU support removed
            return self._orthogonalize_correlated_features_gpu(features, threshold)
        else:
            return self._orthogonalize_correlated_features_cpu_vectorized(features, threshold)

    def _orthogonalize_correlated_features_cpu_vectorized(self,
                                                        features: pd.DataFrame,
                                                        threshold: float = 0.9
                                                        ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """CPU-optimized vectorized correlation analysis."""
        # Vectorized correlation calculation
        corr_matrix = features.corr().abs()
        corr_matrix = corr_matrix.fillna(0.0)
        np.fill_diagonal(corr_matrix.values, 0.0)

        visited = set()
        rotation_metadata: Dict[str, Any] = {}

        # Vectorized group finding
        for column in corr_matrix.columns:
            if column in visited:
                continue

            group = {column}
            stack = [column]

            while stack:
                current = stack.pop()
                visited.add(current)
                # Vectorized partner finding
                strong_partners = corr_matrix.loc[current][corr_matrix.loc[current] > threshold].index.tolist()
                for partner in strong_partners:
                    if partner not in group:
                        group.add(partner)
                        if partner not in visited:
                            stack.append(partner)

            if len(group) < 2:
                continue

            ordered_group = sorted(group, key=lambda name: list(features.columns).index(name))
            subset = features[ordered_group]

            # Vectorized mean calculation and centering
            column_means = subset.mean(axis=0)
            subset_filled = subset.fillna(column_means)
            centered = subset_filled - column_means

            if np.allclose(centered.values, 0.0):
                continue

            try:
                # Vectorized SVD
                _, _, vh = np.linalg.svd(centered.values, full_matrices=False)
            except np.linalg.LinAlgError:
                continue

            rotation_matrix = vh.T
            rotated_values = centered.values @ rotation_matrix

            rotated_df = pd.DataFrame(rotated_values, index=subset.index, columns=ordered_group)
            features.loc[:, ordered_group] = rotated_df

            group_key = "::".join(ordered_group)
            rotation_metadata[group_key] = {
                'columns': ordered_group,
                'means': column_means.to_dict(),
                'rotation_matrix': rotation_matrix.tolist(),
                'method': 'pca',
                'threshold': threshold
            }

        return features, rotation_metadata

    def _orthogonalize_correlated_features_gpu(self,
                                             features: pd.DataFrame,
                                             threshold: float = 0.9
                                             ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """CPU-based correlation analysis and orthogonalization (GPU support removed)."""
        return self._orthogonalize_correlated_features_cpu_vectorized(features, threshold)

    def _create_artifacts(self,
                         transform_router: TransformRouter,
                         lookback_choices: Dict[str, Any],
                         interaction_config: Dict[str, Any],
                         patch_features: Dict[str, pd.Series],
                         rotation_metadata: Dict[str, Any]) -> ArtifactsRegistry:
        """Create artifacts registry."""

        # Transform parameters
        transform_params = {}
        for feature_name, params in transform_router.get_transform_params().items():
            transform_params[feature_name] = {
                'transform_type': 'ewz',  # Simplified
                'params': params,
                'spec_hash': f"transform_{feature_name}_{hash(str(params))}"
            }

        # Lookback choices
        lookback_artifacts = {}
        for family, choice in lookback_choices.items():
            lookback_artifacts[family] = {
                'family': family,
                'selected_lookback': choice.selected_lookback,
                'selection_criteria': 'ic',
                'confidence_score': 0.8,
                'spec_hash': f"lookback_{family}_{choice.selected_lookback}"
            }

        # Interaction configs
        interaction_artifacts = {}
        for interaction_id, config in interaction_config.items():
            interaction_artifacts[interaction_id] = {
                'interaction_id': interaction_id,
                'formula': config.formula,
                'required_fields': config.required_fields,
                'regime_dependent': config.regime_dependent,
                'spec_hash': f"interaction_{interaction_id}"
            }

        # Model artifacts
        model_artifacts = {}
        if patch_features:
            model_artifacts['patch_model'] = {
                'model_type': 'gru',
                'model_object': None,  # Would store actual model
                'training_metadata': {'horizons': self.config.patch_horizons},
                'feature_importance': {},
                'spec_hash': 'patch_model_gru'
            }

        return ArtifactsRegistry(
            transform_params=transform_params,
            lookback_choices=lookback_artifacts,
            interaction_configs=interaction_artifacts,
            model_artifacts=model_artifacts,
            rotation_metadata=rotation_metadata,
            patch_weights=None,
            residual_std=None,
            spec_hash=f"assembly_{hash(str(transform_params))}"
        )

def create_assembly_pipeline(config: AssemblyConfig) -> AssemblyDAG:
    """Create assembly pipeline with configuration."""
    return AssemblyDAG(config)

def run_assembly(bars: pd.DataFrame,
                targets: Optional[Dict[int, pd.Series]] = None,
                config: Optional[AssemblyConfig] = None) -> AssemblyResult:
    """Run complete assembly pipeline."""

    if config is None:
        config = AssemblyConfig()

    pipeline = create_assembly_pipeline(config)
    return pipeline.assemble(bars, targets)
