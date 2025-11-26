"""
Feature Generation Feature Generation Step.

This step generates features from market data.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from src.training.steps.base_step import BaseStep
from src.training.steps.pre_training.includes import BasicFeatureAnalysisInclude
from src.utils.logger import system_logger
from src.utils.tprint import tprint

logger = logging.getLogger(__name__)


class FeatureGenerationFeatureGenerationStep(BaseStep):
    """
    Feature Generation Feature Generation Step.

    Generates features from market data using the unified feature generation system.
    """

    def __init__(self, step_name: str = "feature_generation_feature_generation_step"):
        """Initialize the feature generation step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('FeatureGeneration')
        self.basic_feature_analysis_include = BasicFeatureAnalysisInclude()

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute feature generation.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - execution_mode: 'full', 'light', or 'blank'

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        # Set context with symbol from config to ensure proper versioned store path
        symbol = config.get('symbol', 'UNKNOWN')
        exchange = config.get('exchange', 'binance')
        timeframe = config.get('timeframe', '15m')
        direction = config.get('direction', 'long')
        model = config.get('model', 'analyst')
        
        self.set_context(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            model=model
        )
        
        tprint(f"⚙️ Starting feature generation for {symbol}")

        try:
            # Load data and generate features
            market_data = await self._load_market_data(config)
            features = await self._generate_features(market_data, config)

            # Save generated features as artifact
            timeframe = config.get('timeframe', '15m')
            artifact_name = f'generated_features_{timeframe}'
            features_artifact_path = self._save_artifact(
                data=features,
                artifact_name=artifact_name,
                artifact_type='data',
                metadata={
                    'symbol': config['symbol'],
                    'exchange': config['exchange'],
                    'timeframe': timeframe,
                    'execution_mode': config.get('execution_mode', 'light'),
                    'created_at': datetime.now().isoformat(),
                    'n_features': len(features.columns) if hasattr(features, 'columns') else 0
                }
            )

            artifacts = {
                'generated_features': features_artifact_path
            }

            # Calculate feature statistics for constant features
            constant_features = 0
            constant_feature_names = []
            if hasattr(features, 'columns'):
                for col in features.columns:
                    # More robust constant feature detection
                    col_data = features[col].dropna()  # Remove NaN values first
                    if len(col_data) == 0:
                        # All values are NaN, consider as constant
                        constant_features += 1
                        constant_feature_names.append(col)
                    elif col_data.nunique() <= 1:
                        # Only 1 unique value (excluding NaN)
                        constant_features += 1
                        constant_feature_names.append(col)
                    elif col_data.std() == 0:
                        # Zero standard deviation (all values identical)
                        constant_features += 1
                        constant_feature_names.append(col)
            
            metrics = {
                'n_features_generated': len(features.columns) if hasattr(features, 'columns') else 0,
                'feature_categories': 5,
                'data_rows': len(features),
                'execution_mode': config.get('execution_mode', 'light'),
                'constant_features': constant_features,
                'constant_feature_names': constant_feature_names,
                'success': True
            }

            tprint(f"✅ Feature generation completed: {metrics['n_features_generated']} features")
            
            # Log constant features if any
            if 'constant_features' in metrics and metrics['constant_features'] > 0:
                tprint(f"⚠️  Found {metrics['constant_features']} constant features (zero variance)")
                if 'constant_feature_names' in metrics and metrics['constant_feature_names']:
                    tprint("📋 Constant features:")
                    for i, feature_name in enumerate(metrics['constant_feature_names'][:10], 1):  # Show first 10
                        tprint(f"   {i}. {feature_name}")
                    if len(metrics['constant_feature_names']) > 10:
                        tprint(f"   ... and {len(metrics['constant_feature_names']) - 10} more")
                
                # Diagnostic: Print detailed info about constant features
                if 'constant_feature_names' in metrics and metrics['constant_feature_names']:
                    tprint("🔍 DIAGNOSTIC: Investigating constant features...")
                    for feat_name in metrics['constant_feature_names'][:5]:  # Check first 5
                        feat_data = features[feat_name]
                        tprint(f"  • {feat_name}:")
                        tprint(f"    - Unique values: {feat_data.nunique()}")
                        tprint(f"    - Non-null count: {feat_data.notna().sum()}")
                        if feat_data.nunique() <= 3:  # Show sample values for low-cardinality
                            unique_vals = feat_data.dropna().unique()
                            tprint(f"    - Sample values: {unique_vals[:5]}")
                        tprint(f"    - Std deviation: {feat_data.std():.6f}")
                
                # Optionally remove constant features (configurable)
                remove_constants = config.get('remove_constant_features', True)  # Default: True
                if remove_constants and 'constant_feature_names' in metrics and metrics['constant_feature_names']:
                    tprint(f"🗑️  Removing {len(metrics['constant_feature_names'])} constant features...")
                    features = features.drop(columns=metrics['constant_feature_names'])
                    tprint(f"✅ Removed constant features. Remaining: {len(features.columns)} features")
                    # Update metrics after removal
                    metrics['n_features_generated'] = len(features.columns)
                    metrics['constant_features'] = 0
                    metrics['constant_feature_names'] = []
                    # Re-save artifact with cleaned features
                    tprint("💾 Re-saving artifact with cleaned features...")
                    features_artifact_path = self._save_artifact(
                        data=features,
                        artifact_name=artifact_name,
                        artifact_type='data',
                        metadata={
                            'symbol': config['symbol'],
                            'exchange': config['exchange'],
                            'timeframe': timeframe,
                            'execution_mode': config.get('execution_mode', 'light'),
                            'created_at': datetime.now().isoformat(),
                            'n_features': len(features.columns),
                            'removed_constants': len(metrics.get('constant_feature_names', []))
                        }
                    )
                    artifacts['generated_features'] = features_artifact_path
                    tprint(f"✅ Artifact updated: {features_artifact_path}")
                elif not remove_constants:
                    tprint("ℹ️  Keeping constant features (remove_constant_features=False)")

            features, filter_stats = self._apply_conservative_feature_filters(features, config)
            if filter_stats['total_removed'] > 0:
                tprint(
                    f"🧹 Conservative filtering removed {filter_stats['total_removed']} features "
                    f"(low variance: {len(filter_stats['low_variance_removed'])}, "
                    f"high correlation: {len(filter_stats['high_correlation_removed'])})."
                )
                metrics['n_features_generated'] = len(features.columns)
                metrics['filtering'] = filter_stats
                tprint("💾 Re-saving artifact after conservative filtering...")
                features_artifact_path = self._save_artifact(
                    data=features,
                    artifact_name=artifact_name,
                    artifact_type='data',
                    metadata={
                        'symbol': config['symbol'],
                        'exchange': config['exchange'],
                        'timeframe': timeframe,
                        'execution_mode': config.get('execution_mode', 'light'),
                        'created_at': datetime.now().isoformat(),
                        'n_features': len(features.columns),
                        'low_variance_removed': len(filter_stats['low_variance_removed']),
                        'high_correlation_removed': len(filter_stats['high_correlation_removed'])
                    }
                )
                artifacts['generated_features'] = features_artifact_path

            analysis_include_results = self.basic_feature_analysis_include.run(features, config)
            if analysis_include_results:
                metrics['basic_feature_analysis'] = analysis_include_results.get('metrics', {})
                include_report = analysis_include_results.get('report_path')
                if include_report:
                    artifacts['basic_feature_analysis_report'] = include_report

            # Run baseline predictive check if enabled
            baseline_check_enabled = config.get('run_baseline_check', True)  # Default: enabled
            baseline_check_results = None
            if baseline_check_enabled:
                tprint(f"🔍 Running baseline predictive check...")
                try:
                    baseline_check_results = self._run_baseline_predictive_check(features, config)
                    if baseline_check_results and baseline_check_results.get('success', False):
                        tprint(f"✅ Baseline predictive check completed")
                        metrics['baseline_check'] = baseline_check_results
                    else:
                        tprint(f"⚠️ Baseline predictive check failed or returned no results")
                except Exception as e:
                    tprint(f"⚠️ Baseline predictive check failed: {e}")
                    self.logger.warning(f"Baseline predictive check failed: {e}")
            else:
                tprint(f"ℹ️ Baseline predictive check disabled (run_baseline_check=False)")

            # Generate outcome report
            report_path = self._generate_outcome_report(metrics, artifacts, config, baseline_check_results)
            if report_path:
                tprint(f"📄 Outcome report: {report_path}")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics
            }

        except Exception as e:
            error_msg = f"Feature generation failed: {str(e)}"
            tprint(f"❌ {error_msg}")
            self.logger.error(error_msg)

            return {
                'success': False,
                'artifacts': {},
                'metrics': {},
                'error': error_msg
            }

    async def _load_market_data(self, config: Dict[str, Any]) -> Any:
        """Load market data for feature generation via BaseStep helpers."""
        try:
            # Delegate to BaseStep so execution mode and lookback days are handled centrally
            pipeline_state: Dict[str, Any] = {}
            market_data, _ = self.load_market_data_or_fail(
                config,
                pipeline_state=pipeline_state,
                allow_config_override=True,
            )
            return market_data

        except Exception as e:
            self.logger.error(f"Failed to load market data via BaseStep: {e}")
            return None

    def _load_optimized_lookbacks(self, config: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Load category-level optimized lookbacks from the lookback optimization step."""
        try:
            # Set context to the optimization step to load its artifacts
            symbol = config.get('symbol')
            exchange = config.get('exchange')
            timeframe = config.get('timeframe')
            direction = config.get('direction')
            model = config.get('model')

            from src.utils.artifact_manager import ArtifactManager

            temp_artifact_manager = ArtifactManager(config={})
            temp_artifact_manager.set_context(
                step_name='feature_generation_period_lookback_optimization_step',
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model=model,
            )

            # Load the lookback optimization artifact; this is typically a Parquet DataFrame
            # created from a nested dict containing 'optimized_lookbacks'.
            lookback_result = temp_artifact_manager.get_artifact(
                artifact_name='lookback_optimization',
                artifact_type='data',
                return_path=True,
            )

            lookback_artifact: Any
            lookback_path: Optional[str]
            if isinstance(lookback_result, tuple):
                lookback_artifact, lookback_path = lookback_result
            else:
                lookback_artifact, lookback_path = lookback_result, None

            optimized_lookbacks: Dict[str, int] = {}

            # Case 1: Legacy dict/json artifact
            if isinstance(lookback_artifact, dict):
                data_block = lookback_artifact.get('data', lookback_artifact)
                if not isinstance(data_block, dict):
                    data_block = lookback_artifact

                raw_optimized = data_block.get('optimized_lookbacks', {})
                if isinstance(raw_optimized, dict):
                    for category, value in raw_optimized.items():
                        try:
                            optimized_lookbacks[str(category)] = int(value)
                        except (TypeError, ValueError):
                            continue

            # Case 2: Parquet DataFrame produced from nested dict via json_normalize
            elif isinstance(lookback_artifact, pd.DataFrame):
                df = lookback_artifact
                if not df.empty:
                    row = df.iloc[0]
                    for col in df.columns:
                        if not isinstance(col, str):
                            continue
                        if not col.startswith('optimized_lookbacks.'):
                            continue

                        category_key = col.split('optimized_lookbacks.', 1)[1]
                        value = row[col]
                        # Skip NaNs or missing values
                        if pd.isna(value):
                            continue
                        try:
                            optimized_lookbacks[str(category_key)] = int(value)
                        except (TypeError, ValueError):
                            continue

            if optimized_lookbacks:
                if lookback_path:
                    self.logger.info(f"lookback_optimization data artifact path: {lookback_path}")

                tprint("✅ Loaded optimized lookback periods from optimization step:")
                for category, lookback in optimized_lookbacks.items():
                    tprint(f"   • {category}: {lookback}")

                # Map category names from optimization format to FeatureBank format
                # Optimization uses: 'momentum_features', 'trend_features', etc.
                # FeatureBank uses: 'momentum', 'trend', etc.
                mapped_lookbacks: Dict[str, int] = {}
                for category, lookback in optimized_lookbacks.items():
                    clean_category = category.replace('_features', '')
                    mapped_lookbacks[clean_category] = int(lookback)

                return mapped_lookbacks

            if lookback_path:
                self.logger.info(f"lookback_optimization data artifact search path: {lookback_path}")
            tprint("ℹ️ No optimized lookback periods found in artifact - will use default lookback periods")
            return None

        except Exception as e:
            self.logger.debug(f"Could not load optimized lookbacks: {e}")
            tprint("ℹ️ Could not load optimized lookbacks - will use default lookback periods")
            return None

    def _load_per_feature_lookbacks(self, config: Dict[str, Any]) -> Optional[Dict[str, List[int]]]:
        """Load per-feature optimized lookbacks (final feature names → [lb1, lb2, lb3])."""
        try:
            symbol = config.get('symbol')
            exchange = config.get('exchange')
            timeframe = config.get('timeframe')
            direction = config.get('direction')
            model = config.get('model')

            from src.utils.artifact_manager import ArtifactManager

            temp_artifact_manager = ArtifactManager(config={})
            temp_artifact_manager.set_context(
                step_name='feature_generation_period_lookback_optimization_step',
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model=model,
            )

            # Preferred path: dedicated metadata artifact with the per-feature mapping
            metadata_mapping = None
            metadata_path: Optional[str] = None
            try:
                metadata_result = temp_artifact_manager.get_artifact(
                    artifact_name='per_feature_lookbacks',
                    artifact_type='metadata',
                    return_path=True,
                )
                if isinstance(metadata_result, tuple):
                    metadata_mapping, metadata_path = metadata_result
                else:
                    metadata_mapping = metadata_result
            except Exception as inner_exc:
                self.logger.debug(f"Could not load per-feature lookbacks metadata: {inner_exc}")
                metadata_mapping = None

            if isinstance(metadata_mapping, dict) and metadata_mapping:
                tprint(
                    f"✅ Loaded per-feature lookbacks from metadata for {len(metadata_mapping)} features"
                )
                if metadata_path:
                    self.logger.info(f"per_feature_lookbacks metadata path: {metadata_path}")
                return {
                    str(name): [int(lb) for lb in lbs]
                    for name, lbs in metadata_mapping.items()
                    if isinstance(lbs, (list, tuple)) and lbs
                }
            else:
                if metadata_mapping is None:
                    tprint("ℹ️ per_feature_lookbacks metadata artifact not found or could not be loaded")
                elif not isinstance(metadata_mapping, dict):
                    tprint(
                        f"ℹ️ per_feature_lookbacks metadata artifact loaded but is not a dict "
                        f"(type={type(metadata_mapping).__name__})"
                    )
                elif not metadata_mapping:
                    tprint("ℹ️ per_feature_lookbacks metadata artifact is an empty dict")
                if metadata_path:
                    self.logger.info(f"per_feature_lookbacks metadata search path: {metadata_path}")

            # Backwards-compatible path: embedded mapping inside the lookback_optimization data artifact
            lookback_result = temp_artifact_manager.get_artifact(
                artifact_name='lookback_optimization',
                artifact_type='data',
                return_path=True,
            )

            lookback_artifact: Any
            lookback_path: Optional[str]
            if isinstance(lookback_result, tuple):
                lookback_artifact, lookback_path = lookback_result
            else:
                lookback_artifact, lookback_path = lookback_result, None

            if lookback_artifact is not None and isinstance(lookback_artifact, dict):
                data_block = lookback_artifact.get('data', lookback_artifact)
                if not isinstance(data_block, dict):
                    data_block = lookback_artifact

                per_feature_lookbacks = data_block.get('per_feature_lookbacks', {})
                if per_feature_lookbacks:
                    tprint(
                        f"✅ Loaded per-feature lookbacks for {len(per_feature_lookbacks)} features "
                        f"from optimization step data artifact"
                    )
                    if lookback_path:
                        self.logger.info(f"lookback_optimization data artifact path: {lookback_path}")
                    return {
                        str(name): [int(lb) for lb in lbs]
                        for name, lbs in per_feature_lookbacks.items()
                        if isinstance(lbs, (list, tuple)) and lbs
                    }

                tprint("ℹ️ No per-feature lookbacks found in lookback_optimization artifact")
                return None

            if lookback_path:
                self.logger.info(f"lookback_optimization data artifact search path: {lookback_path}")
            tprint("ℹ️ No lookback optimization artifacts found for per-feature lookbacks")
            return None

        except Exception as e:
            self.logger.debug(f"Could not load per-feature lookbacks: {e}")
            tprint("ℹ️ Could not load per-feature lookbacks - will fall back to category/default lookbacks")
            return None

    async def _generate_features(self, market_data: Any, config: Dict[str, Any]) -> Any:
        """Generate features from market data using the unified feature generation system."""
        from src.feature_generation.core.feature_bank import FeatureBank

        if market_data is None:
            return pd.DataFrame()

        tprint(f"🔧 Generating features using FeatureBank")

        # Load optimized lookback periods if available
        optimized_lookbacks = self._load_optimized_lookbacks(config)
        per_feature_lookbacks = self._load_per_feature_lookbacks(config)

        # Use the unified feature generation system
        feature_bank = FeatureBank()

        # Get all registered feature categories
        feature_categories = [
            'returns',
            'momentum',
            'volume',
            'volatility',
            'trend',
            'oscillator',
            'support_resistance',
            'candlestick_pattern',
            'entropy',
            'acceleration',
            'advanced_statistical',
            'spectral_wavelet',
            'time'  # Time features: hour_sin, hour_cos, day_of_week_sin, day_of_week_cos
        ]

        # Fix FeatureBank bug by disabling optimized pipeline
        # The optimized pipeline has a bug where all features get identical values
        tprint(f"🔧 Using FeatureBank with standard generation (fixing optimized pipeline bug)")

        try:
            # Generate features using FeatureBank but with optimized pipeline disabled
            # Add minimum data validation to prevent all-NaN columns
            data_length = len(market_data)

            # Determine lookback periods to use
            # Priority: 1) Optimized lookbacks, 2) Data-based adjustment, 3) Default
            if optimized_lookbacks:
                # Use optimized lookback periods from the optimization step
                tprint("✅ Using optimized lookback periods for feature generation")

                # Create kwargs with category-specific and per-feature lookback periods
                generation_kwargs: Dict[str, Any] = {
                    'optimized_lookbacks': optimized_lookbacks,
                    'use_optimized_pipeline': False,
                    'progressive_loading': True,
                    'auto_normalize': False,
                }

                if per_feature_lookbacks:
                    generation_kwargs['per_feature_lookbacks'] = per_feature_lookbacks
                    tprint(
                        f"   • Applying per-feature lookbacks for {len(per_feature_lookbacks)} features "
                        f"(keys treated as final feature names)"
                    )

                # For each category, pass the specific lookback period
                for category in feature_categories:
                    if category in optimized_lookbacks:
                        generation_kwargs[f'{category}_lookback'] = optimized_lookbacks[category]
                        tprint(f"   • {category}: using lookback period {optimized_lookbacks[category]}")

                generated_features = feature_bank.generate_features(
                    data=market_data,
                    categories=feature_categories,
                    **generation_kwargs,
                )
            elif data_length < 100:
                # Adjust lookback periods based on available data to prevent all-NaN columns
                max_lookback = min(20, data_length // 3) if data_length < 60 else 20

                if data_length < 50:
                    tprint(f"⚠️ Warning: Only {data_length} rows of data available. Some indicators may produce all-NaN columns.")
                    tprint("💡 Consider increasing data size or using shorter lookback periods")
                else:
                    tprint(f"⚠️ Limited data: {data_length} rows. Adjusting lookback periods to prevent all-NaN columns.")
                    tprint(f"🔧 Using maximum lookback period of {max_lookback} instead of 20")

                # Create a custom configuration with adjusted lookback periods
                custom_config = {
                    'lookback_periods': {
                        'short': min(5, data_length // 10),
                        'medium': min(10, data_length // 6),
                        'long': max_lookback
                    }
                }

                generated_features = feature_bank.generate_features(
                    data=market_data,
                    categories=feature_categories,
                    use_optimized_pipeline=False,  # Disable optimized pipeline to fix identical values bug
                    progressive_loading=True,
                    auto_normalize=False,  # Disable automatic normalization to prevent zero-mean issue
                    custom_config=custom_config  # Pass custom configuration
                )
            else:
                # Use default lookback periods
                tprint(f"ℹ️ Using default lookback periods for feature generation")
                generated_features = feature_bank.generate_features(
                    data=market_data,
                    categories=feature_categories,
                    use_optimized_pipeline=False,  # Disable optimized pipeline to fix identical values bug
                    progressive_loading=True,
                    auto_normalize=False  # Disable automatic normalization to prevent zero-mean issue
                )
            
            tprint(f"✅ Generated {len(generated_features.columns)} features from FeatureBank (standard mode)")
            return generated_features
            
        except Exception as e:
            self.logger.warning(f"FeatureBank generation failed: {e}, falling back to simple features")
            
            # Fallback to simple feature generation
            import pandas as pd
            import numpy as np
            
            features = pd.DataFrame(index=market_data.index)
            
            # Generate comprehensive features manually
            features = self._add_basic_features(features, market_data)
            
            # Fill NaN values
            features = features.fillna(0)
            
            tprint(f"✅ Generated {len(features.columns)} features using fallback simple generation")
            return features

    def _run_baseline_predictive_check(self, features: pd.DataFrame, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run baseline predictive check on generated features."""
        try:
            from src.training.steps.pre_training.baseline_predictive_check import BaselinePredictiveCheck
            from pathlib import Path

            # Need a target variable - prefer binary_label if present, then fused/simplified targets,
            # otherwise create a simple return-based target.

            # First, if labeled_data was saved by feature_generation_labeling_integration_step,
            # merge those targets into the local features DataFrame using the same artifact
            # resolution and alignment strategy as the period lookback optimization step.
            merged = self._attach_labels_for_baseline(features, config)
            if merged is not None:
                features = merged

            target = None
            direction = str(config.get('direction', 'long')).lower()

            # 0) Prefer primary binary meta-label when available
            if 'binary_label' in features.columns:
                target = features['binary_label']

            # 1) Prefer fused targets if present
            if target is None and ('target_long_fused' in features.columns or 'target_short_fused' in features.columns):
                if 'short' in direction and 'target_short_fused' in features.columns:
                    target = features['target_short_fused']
                elif 'long' in direction and 'target_long_fused' in features.columns:
                    target = features['target_long_fused']
                elif 'target_long_fused' in features.columns:
                    target = features['target_long_fused']
                elif 'target_short_fused' in features.columns:
                    target = features['target_short_fused']

            # 2) Fall back to simplified long/short targets
            if target is None and ('target_long' in features.columns or 'target_short' in features.columns):
                if 'short' in direction and 'target_short' in features.columns:
                    target = features['target_short']
                elif 'long' in direction and 'target_long' in features.columns:
                    target = features['target_long']
                elif 'target_long' in features.columns:
                    target = features['target_long']
                elif 'target_short' in features.columns:
                    target = features['target_short']

            # 3) If no explicit targets exist, create a simple return-based target
            if target is None:
                if 'close' in features.columns:
                    # Create a simple target: next period return
                    target = features['close'].pct_change().shift(-1)
                elif 'returns_features_returns' in features.columns:
                    # Use returns feature as target
                    target = features['returns_features_returns'].shift(-1)
                else:
                    # Try to find any column with 'return' in name
                    return_cols = [col for col in features.columns if 'return' in col.lower()]
                    if return_cols:
                        target = features[return_cols[0]].shift(-1)
                    else:
                        tprint("⚠️ No suitable target found for baseline check, skipping")
                        return None

            # Drop NaNs from target
            target = target.dropna()
            if target.empty:
                tprint("⚠️ Target series is empty after NaN removal, skipping baseline check")
                return None

            # Remove target-like columns from features
            target_like_cols = [
                'target_long_fused', 'target_short_fused',
                'target_long', 'target_short',
                'target', 'label',
                'close', 'returns_features_returns',
            ]
            feature_cols = [col for col in features.columns if col not in target_like_cols]
            X = features[feature_cols]

            # Run the check with max 400 features
            tprint(f"🔍 Running baseline check on {len(X.columns)} features (sampling max 400)...")
            checker = BaselinePredictiveCheck(max_features=400, random_state=42)
            results = checker.run_check(X, target)

            # Save CSVs to outcomes directory (univariate + small multivariate baseline)
            if results.get('success', False):
                outcomes_dir = Path('outcomes')
                outcomes_dir.mkdir(exist_ok=True)

                csv_path = checker.save_results_to_csv(outcomes_dir, filename_prefix="baseline_check_feature_generation")
                if csv_path:
                    tprint(f"📊 Baseline check CSV saved: {csv_path}")
                    results['csv_path'] = csv_path

                multivariate_csv_path = checker.save_multivariate_results_to_csv(
                    outcomes_dir,
                    filename_prefix="multivariate_baseline_feature_generation",
                )
                if multivariate_csv_path:
                    tprint(f"📊 Multivariate baseline CSV saved: {multivariate_csv_path}")
                    results['multivariate_csv_path'] = multivariate_csv_path

            return results

        except Exception as e:
            self.logger.error(f"Baseline predictive check failed: {e}", exc_info=True)
            return None

    def _attach_labels_for_baseline(self, features: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Attach labeled targets to features for baseline checks.

        Loads labeled_data_{symbol}_{timeframe} from the
        feature_generation_labeling_integration_step store and merges it with
        the provided features using the same context and alignment heuristics as
        FeatureGenerationPeriodLookbackOptimizationStep._load_generated_features.
        Returns a merged DataFrame on success, or None if labels cannot be
        loaded.
        """

        try:
            import pandas as pd

            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            model = config.get('model', 'analyst')

            original_step_name = self.artifact_manager._current_step_name
            self.artifact_manager.set_context(
                step_name='feature_generation_labeling_integration_step',
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model=model,
                datetime=datetime.now(),
            )

            labeled_data = self.artifact_manager.get_artifact(
                artifact_name=f'labeled_data_{symbol}_{timeframe}',
                artifact_type='data',
            )

            # Restore original context regardless of load success
            self.artifact_manager.set_context(
                step_name=original_step_name,
                datetime=datetime.now(),
            )

            if labeled_data is None:
                return None

            labeled_data = labeled_data if isinstance(labeled_data, pd.DataFrame) else pd.DataFrame(labeled_data)
            if labeled_data.empty:
                return None

            # Identify target columns from labeled data
            priority_target_columns = [
                'binary_label',
                'target_long_fused',
                'target_short_fused',
                'target_long',
                'target_short',
                'price_target_vol_normalized',
                'target_sample_weight',
                'meta_probability',
                'r_multiple',
            ]
            target_columns = [col for col in priority_target_columns if col in labeled_data.columns]
            target_columns += [
                col for col in labeled_data.columns
                if 'target' in str(col).lower() and col not in target_columns
            ]
            if not target_columns:
                target_columns = labeled_data.columns.tolist()

            features_df = features.copy()
            targets_df = labeled_data[target_columns].copy()

            # Heuristic 1: if lengths match, align by position
            if len(features_df) == len(targets_df) and len(features_df) > 0:
                features_df = features_df.reset_index(drop=True)
                targets_df = targets_df.reset_index(drop=True)
                merged = pd.concat([features_df, targets_df], axis=1)
                return merged

            # Heuristic 2: try to build a datetime index on both sides
            if not isinstance(features_df.index, pd.DatetimeIndex):
                try:
                    features_idx = pd.Index(features_df.index)
                    if features_idx.dtype == object:
                        decoded = features_idx.astype(str).str.replace("^b'|'$", "", regex=True)
                        features_df.index = pd.to_datetime(decoded, errors='coerce')
                    else:
                        features_df.index = pd.to_datetime(features_idx, errors='coerce')
                except Exception:
                    pass

            if not isinstance(targets_df.index, pd.DatetimeIndex):
                for time_col in ['open_time', 'close_time']:
                    if time_col in labeled_data.columns:
                        try:
                            targets_df.index = pd.to_datetime(labeled_data[time_col], errors='coerce')
                            break
                        except Exception:
                            continue

            if isinstance(features_df.index, pd.DatetimeIndex):
                features_df = features_df[features_df.index.notna()]
            if isinstance(targets_df.index, pd.DatetimeIndex):
                targets_df = targets_df[targets_df.index.notna()]

            merged = features_df.join(targets_df, how='inner')
            if merged.empty:
                return None

            return merged

        except Exception:
            return None

    def _add_basic_features(self, features: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Add basic features as fallback when FeatureBank doesn't generate enough."""
        import pandas as pd
        import numpy as np
        
        tprint(f"🔧 Adding basic features as fallback...")
        
        # Price-based features (returns category)
        if 'close' in market_data.columns:
            features['returns_features_returns'] = market_data['close'].pct_change()
            features['returns_features_price_ma_5'] = market_data['close'].rolling(5).mean()
            features['returns_features_price_std_5'] = market_data['close'].rolling(5).std()
            features['returns_features_price_ma_10'] = market_data['close'].rolling(10).mean()
            features['returns_features_price_std_10'] = market_data['close'].rolling(10).std()
            features['returns_features_price_ma_20'] = market_data['close'].rolling(20).mean()
            features['returns_features_price_std_20'] = market_data['close'].rolling(20).std()
            
            # RSI calculation (oscillator category)
            delta = market_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['oscillator_features_rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD (oscillator category)
            ema_12 = market_data['close'].ewm(span=12).mean()
            ema_26 = market_data['close'].ewm(span=26).mean()
            features['oscillator_features_macd'] = ema_12 - ema_26
            features['oscillator_features_macd_signal'] = features['oscillator_features_macd'].ewm(span=9).mean()
            features['oscillator_features_macd_histogram'] = features['oscillator_features_macd'] - features['oscillator_features_macd_signal']
        
        # Volume-based features (volume category)
        if 'volume' in market_data.columns:
            features['volume_features_volume_ma_5'] = market_data['volume'].rolling(5).mean()
            features['volume_features_volume_std_5'] = market_data['volume'].rolling(5).std()
            features['volume_features_volume_ma_10'] = market_data['volume'].rolling(10).mean()
            features['volume_features_volume_std_10'] = market_data['volume'].rolling(10).std()
            features['volume_features_volume_ma_20'] = market_data['volume'].rolling(20).mean()
            features['volume_features_volume_std_20'] = market_data['volume'].rolling(20).std()
            
            # Volume-price relationship
            features['volume_features_volume_price_trend'] = market_data['volume'] * market_data['close'].pct_change()
            features['volume_features_volume_sma_ratio'] = market_data['volume'] / market_data['volume'].rolling(20).mean()
        
        # High-Low features (volatility category)
        if 'high' in market_data.columns and 'low' in market_data.columns:
            features['volatility_features_hl_range'] = market_data['high'] - market_data['low']
            features['volatility_features_hl_range_pct'] = features['volatility_features_hl_range'] / market_data['close']
            features['volatility_features_hl_ma_5'] = features['volatility_features_hl_range'].rolling(5).mean()
            features['volatility_features_hl_ma_10'] = features['volatility_features_hl_range'].rolling(10).mean()
        
        # Open-Close features (returns category)
        if 'open' in market_data.columns and 'close' in market_data.columns:
            features['returns_features_oc_return'] = (market_data['close'] - market_data['open']) / market_data['open']
            features['returns_features_oc_abs'] = abs(features['returns_features_oc_return'])
            features['returns_features_oc_ma_5'] = features['returns_features_oc_return'].rolling(5).mean()
            features['returns_features_oc_ma_10'] = features['returns_features_oc_return'].rolling(10).mean()
        
        # Volatility features (volatility category)
        if 'close' in market_data.columns:
            features['volatility_features_volatility_5'] = market_data['close'].pct_change().rolling(5).std()
            features['volatility_features_volatility_10'] = market_data['close'].pct_change().rolling(10).std()
            features['volatility_features_volatility_20'] = market_data['close'].pct_change().rolling(20).std()
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = market_data['close'].rolling(bb_period).mean()
            bb_std_val = market_data['close'].rolling(bb_period).std()
            features['volatility_features_bb_upper'] = bb_middle + (bb_std_val * bb_std)
            features['volatility_features_bb_lower'] = bb_middle - (bb_std_val * bb_std)
            features['volatility_features_bb_width'] = features['volatility_features_bb_upper'] - features['volatility_features_bb_lower']
            features['volatility_features_bb_position'] = (market_data['close'] - features['volatility_features_bb_lower']) / features['volatility_features_bb_width']
        
        # Momentum features (momentum category)
        if 'close' in market_data.columns:
            features['momentum_features_momentum_5'] = market_data['close'] / market_data['close'].shift(5) - 1
            features['momentum_features_momentum_10'] = market_data['close'] / market_data['close'].shift(10) - 1
            features['momentum_features_momentum_20'] = market_data['close'] / market_data['close'].shift(20) - 1
            
            # Rate of Change
            features['momentum_features_roc_5'] = market_data['close'].pct_change(5)
            features['momentum_features_roc_10'] = market_data['close'].pct_change(10)
            features['momentum_features_roc_20'] = market_data['close'].pct_change(20)
        
        # Trend features (trend category)
        if 'close' in market_data.columns:
            # Simple Moving Average crossovers
            sma_5 = market_data['close'].rolling(5).mean()
            sma_10 = market_data['close'].rolling(10).mean()
            sma_20 = market_data['close'].rolling(20).mean()
            
            features['trend_features_sma_5_10_diff'] = sma_5 - sma_10
            features['trend_features_sma_10_20_diff'] = sma_10 - sma_20
            features['trend_features_sma_5_20_diff'] = sma_5 - sma_20
            
            # Trend strength
            features['trend_features_trend_strength_5'] = (market_data['close'] - sma_5) / sma_5
            features['trend_features_trend_strength_10'] = (market_data['close'] - sma_10) / sma_10
            features['trend_features_trend_strength_20'] = (market_data['close'] - sma_20) / sma_20
        
        # Add some basic advanced_statistical features
        if 'close' in market_data.columns:
            # Skewness and Kurtosis
            features['advanced_statistical_features_skewness_20'] = market_data['close'].pct_change().rolling(20).skew()
            features['advanced_statistical_features_kurtosis_20'] = market_data['close'].pct_change().rolling(20).kurt()
            
            # Hurst-like measure (simplified)
            returns = market_data['close'].pct_change().dropna()
            if len(returns) > 20:
                features['advanced_statistical_features_hurst_like'] = returns.rolling(20).apply(lambda x: np.log(np.var(x)) / np.log(len(x)) if len(x) > 1 else 0)
        
        # Add some basic support_resistance features
        if 'high' in market_data.columns and 'low' in market_data.columns:
            # Simple support/resistance levels
            features['support_resistance_features_resistance_20'] = market_data['high'].rolling(20).max()
            features['support_resistance_features_support_20'] = market_data['low'].rolling(20).min()
            features['support_resistance_features_sr_position'] = (market_data['close'] - features['support_resistance_features_support_20']) / (features['support_resistance_features_resistance_20'] - features['support_resistance_features_support_20'])
        
        tprint(f"✅ Added {len(features.columns)} basic features as fallback")
        return features

    def _apply_conservative_feature_filters(
        self,
        features: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Drop low-variance and near-duplicate features in a conservative manner."""
        stats = {
            'low_variance_removed': [],
            'high_correlation_removed': [],
            'high_correlation_pairs': [],
            'total_removed': 0
        }

        if features is None or features.empty:
            return features, stats

        numeric_features = features.select_dtypes(include=[np.number])
        if numeric_features.empty:
            return features, stats

        low_variance_threshold = float(config.get('low_variance_threshold', 1e-4))
        high_corr_threshold = float(config.get('high_correlation_threshold', 0.999))
        correlation_analysis_max_features = int(config.get('correlation_analysis_max_features', 400))
        protected_features = set(config.get('protected_features', []))

        drop_columns = set()

        # Low-variance filtering
        try:
            variances = numeric_features.var()
            low_variance_cols = variances[variances <= low_variance_threshold].index.tolist()
            stats['low_variance_removed'] = low_variance_cols
            drop_columns.update(low_variance_cols)
        except Exception as exc:
            self.logger.debug(f"Low variance filtering skipped: {exc}")

        # High-correlation filtering (conservative)
        try:
            candidates = [col for col in numeric_features.columns if col not in protected_features]
            if len(candidates) > correlation_analysis_max_features:
                candidates = candidates[:correlation_analysis_max_features]

            sample = numeric_features[candidates].copy()
            sample = sample.dropna(axis=0, how='any')

            if len(sample.columns) > 1 and not sample.empty:
                corr_matrix = sample.corr().abs()
                cols = list(corr_matrix.columns)
                for idx, col in enumerate(cols):
                    if col in drop_columns:
                        continue
                    for jdx in range(idx + 1, len(cols)):
                        other = cols[jdx]
                        if other in drop_columns:
                            continue
                        corr_value = corr_matrix.iat[idx, jdx]
                        if corr_value >= high_corr_threshold:
                            drop_columns.add(other)
                            stats['high_correlation_removed'].append(other)
                            stats['high_correlation_pairs'].append((col, other, float(corr_value)))
            # Limit stored pairs to keep report compact
            stats['high_correlation_pairs'] = stats['high_correlation_pairs'][:20]
        except Exception as exc:
            self.logger.debug(f"High correlation filtering skipped: {exc}")

        if not drop_columns:
            return features, stats

        retained_columns = [col for col in features.columns if col not in drop_columns]
        filtered_features = features[retained_columns]
        stats['total_removed'] = len(drop_columns)
        return filtered_features, stats

    def _generate_outcome_report(self, metrics: Dict[str, Any], artifacts: Dict[str, Any], config: Dict[str, Any], baseline_check_results: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Generate comprehensive outcome report in markdown format."""
        try:
            from pathlib import Path
            
            # Create outcomes directory if it doesn't exist
            outcomes_dir = Path('outcomes')
            outcomes_dir.mkdir(exist_ok=True)
            
            # Generate report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"{self.step_name}_report_{timestamp}.md"
            report_path = outcomes_dir / report_filename
            
            # Load the generated features to get detailed information
            features = None
            if 'generated_features' in artifacts:
                try:
                    artifact_path = artifacts['generated_features']
                    if isinstance(artifact_path, str) and Path(artifact_path).exists():
                        import pandas as pd
                        features = pd.read_parquet(artifact_path)
                except Exception as e:
                    self.logger.warning(f"Could not load features for report: {e}")
            
            # Generate markdown report
            with open(report_path, 'w') as f:
                f.write(f"# Feature Generation Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Step:** {self.step_name}\n\n")
                
                f.write("## Configuration\n\n")
                f.write(f"- **Symbol:** {config.get('symbol', 'N/A')}\n")
                f.write(f"- **Exchange:** {config.get('exchange', 'N/A')}\n")
                f.write(f"- **Timeframe:** {config.get('timeframe', 'N/A')}\n")
                f.write(f"- **Execution Mode:** {config.get('execution_mode', 'N/A')}\n\n")
                
                f.write("## Summary\n\n")
                f.write(f"✅ **Successfully generated {metrics.get('n_features_generated', 0)} features** ")
                f.write(f"from {metrics.get('data_rows', 0):,} rows of data.\n\n")
                
                if features is not None:
                    f.write("## Feature Statistics\n\n")
                    f.write(f"- **Total Features:** {len(features.columns)}\n")
                    f.write(f"- **Data Samples:** {len(features):,}\n")
                    f.write(f"- **Memory Usage:** {features.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n")
                    f.write(f"- **Missing Values:** {features.isnull().sum().sum()}\n")
                    f.write(f"- **Missing Value %:** {(features.isnull().sum().sum() / (len(features) * len(features.columns))) * 100:.2f}%\n\n")
                    
                    # Enhanced feature analysis
                    f.write("## Comprehensive Feature Analysis\n\n")
                    
                    # Calculate detailed statistics for each feature
                    feature_stats = self._calculate_comprehensive_feature_stats(features)
                    
                    # Feature quality metrics
                    f.write("### Feature Quality Metrics\n\n")
                    f.write(f"| Metric | Value |\n")
                    f.write(f"|--------|-------|\n")
                    f.write(f"| High Quality Features (>0.7 score) | {feature_stats['high_quality_count']} |\n")
                    f.write(f"| Medium Quality Features (0.4-0.7) | {feature_stats['medium_quality_count']} |\n")
                    f.write(f"| Low Quality Features (<0.4) | {feature_stats['low_quality_count']} |\n")
                    f.write(f"| Constant Features | {feature_stats['constant_features']} |\n")
                    f.write(f"| Highly Correlated Pairs | {feature_stats['high_correlation_pairs']} |\n")
                    f.write(f"| Average Correlation | {feature_stats['avg_correlation']:.3f} |\n")
                    f.write(f"| Feature Stability Score | {feature_stats['stability_score']:.3f} |\n\n")
                    
                    # List constant features if any
                    if feature_stats.get('constant_feature_names'):
                        f.write("### Constant Features (Zero Variance)\n\n")
                        f.write("The following features have constant values across all data points and should be removed:\n\n")
                        for i, feature_name in enumerate(feature_stats['constant_feature_names'], 1):
                            f.write(f"{i}. `{feature_name}`\n")
                        f.write("\n")
                    
                    # Top performing features
                    f.write("### Top 10 Performing Features\n\n")
                    f.write("| Rank | Feature | Quality Score | Correlation | Stability | Information |\n")
                    f.write("|------|---------|---------------|-------------|-----------|-------------|\n")
                    for i, (feature, stats) in enumerate(feature_stats['top_features'][:10], 1):
                        f.write(f"| {i} | `{feature}` | {stats['quality_score']:.3f} | {stats['correlation']:.3f} | {stats['stability']:.3f} | {stats['information']:.3f} |\n")
                    f.write("\n")
                    
                    # Feature distribution analysis
                    f.write("### Feature Distribution Analysis\n\n")
                    f.write(f"| Statistic | Value |\n")
                    f.write(f"|-----------|-------|\n")
                    f.write(f"| Mean Quality Score | {feature_stats['quality_distribution']['mean']:.3f} |\n")
                    f.write(f"| Median Quality Score | {feature_stats['quality_distribution']['median']:.3f} |\n")
                    f.write(f"| Std Quality Score | {feature_stats['quality_distribution']['std']:.3f} |\n")
                    f.write(f"| Min Quality Score | {feature_stats['quality_distribution']['min']:.3f} |\n")
                    f.write(f"| Max Quality Score | {feature_stats['quality_distribution']['max']:.3f} |\n\n")
                    
                    # Feature redundancy analysis
                    f.write("### Feature Redundancy Analysis\n\n")
                    f.write(f"| Metric | Value |\n")
                    f.write(f"|--------|-------|\n")
                    f.write(f"| Redundant Feature Pairs | {feature_stats['redundant_pairs']} |\n")
                    f.write(f"| Redundancy Rate | {feature_stats['redundancy_rate']:.1%} |\n")
                    f.write(f"| Unique Features | {feature_stats['unique_features']} |\n")
                    f.write(f"| Redundancy Score | {feature_stats['redundancy_score']:.3f} |\n\n")
                    
                    # Feature stability analysis
                    f.write("### Feature Stability Analysis\n\n")
                    f.write(f"| Metric | Value |\n")
                    f.write(f"|--------|-------|\n")
                    f.write(f"| Stable Features (>0.8) | {feature_stats['stable_features']} |\n")
                    f.write(f"| Moderately Stable (0.5-0.8) | {feature_stats['moderate_stable']} |\n")
                    f.write(f"| Unstable Features (<0.5) | {feature_stats['unstable_features']} |\n")
                    f.write(f"| Average Stability | {feature_stats['avg_stability']:.3f} |\n\n")
                    
                    # Feature information content
                    f.write("### Feature Information Content\n\n")
                    f.write(f"| Metric | Value |\n")
                    f.write(f"|--------|-------|\n")
                    f.write(f"| High Information (>0.7) | {feature_stats['high_info_features']} |\n")
                    f.write(f"| Medium Information (0.4-0.7) | {feature_stats['medium_info_features']} |\n")
                    f.write(f"| Low Information (<0.4) | {feature_stats['low_info_features']} |\n")
                    f.write(f"| Average Information | {feature_stats['avg_information']:.3f} |\n\n")
                    
                    # Feature recommendations
                    f.write("### Feature Recommendations\n\n")
                    f.write("#### Features to Keep (High Quality)\n")
                    for feature in feature_stats['recommendations']['keep']:
                        f.write(f"- `{feature}`\n")
                    f.write("\n")
                    
                    f.write("#### Features to Consider Removing (Low Quality)\n")
                    for feature in feature_stats['recommendations']['remove']:
                        f.write(f"- `{feature}`\n")
                    f.write("\n")
                    
                    f.write("#### Features to Investigate (Medium Quality)\n")
                    for feature in feature_stats['recommendations']['investigate']:
                        f.write(f"- `{feature}`\n")
                    f.write("\n")
                    
                    # List feature categories if available
                    feature_categories = {
                        'returns': [col for col in features.columns if any(x in col.lower() for x in ['return', 'pct_change', 'ret'])],
                        'momentum': [col for col in features.columns if any(x in col.lower() for x in ['momentum', 'rsi', 'roc', 'stoch'])],
                        'volume': [col for col in features.columns if any(x in col.lower() for x in ['volume', 'vol'])],
                        'volatility': [col for col in features.columns if any(x in col.lower() for x in ['volatil', 'std', 'atr', 'bb'])],
                        'trend': [col for col in features.columns if any(x in col.lower() for x in ['ma', 'sma', 'ema', 'trend', 'adx'])],
                        'oscillator': [col for col in features.columns if any(x in col.lower() for x in ['osc', 'macd', 'signal'])],
                        'support_resistance': [col for col in features.columns if any(x in col.lower() for x in ['sr', 'support', 'resistance', 'pivot'])],
                        'candlestick': [col for col in features.columns if any(x in col.lower() for x in ['candle', 'pattern', 'doji', 'hammer'])],
                        'entropy': [col for col in features.columns if any(x in col.lower() for x in ['entropy', 'shannon'])],
                        'acceleration': [col for col in features.columns if any(x in col.lower() for x in ['accel', 'velocity'])]
                    }
                    
                    f.write("## Feature Categories\n\n")
                    for category, cols in feature_categories.items():
                        if cols:
                            f.write(f"### {category.capitalize()} ({len(cols)} features)\n\n")
                            # Show first 5 features as examples
                            example_features = cols[:5]
                            for feat in example_features:
                                f.write(f"- `{feat}`\n")
                            if len(cols) > 5:
                                f.write(f"- ... and {len(cols) - 5} more\n")
                            f.write("\n")
                    
                    f.write("## Data Quality\n\n")
                    f.write(f"| Metric | Value |\n")
                    f.write(f"|--------|-------|\n")
                    f.write(f"| Total Columns | {len(features.columns)} |\n")
                    f.write(f"| Total Rows | {len(features):,} |\n")
                    f.write(f"| Non-Null Values | {features.notna().sum().sum():,} |\n")
                    f.write(f"| Null Values | {features.isnull().sum().sum():,} |\n")
                    f.write(f"| Memory Usage (MB) | {features.memory_usage(deep=True).sum() / 1024 / 1024:.2f} |\n\n")
                
                f.write("## Artifacts\n\n")
                for artifact_name, artifact_path in artifacts.items():
                    file_size = Path(artifact_path).stat().st_size / 1024 if Path(artifact_path).exists() else 0
                    f.write(f"### {artifact_name}\n\n")
                    f.write(f"**Path:** `{artifact_path}`\n")
                    f.write(f"**Size:** {file_size:.2f} KB\n\n")
                
                # Add baseline predictive check results if available
                if baseline_check_results and baseline_check_results.get('success', False):
                    from src.training.steps.pre_training.baseline_predictive_check import BaselinePredictiveCheck

                    # Create a temporary checker to format results
                    temp_checker = BaselinePredictiveCheck()
                    temp_checker.results = baseline_check_results

                    # Add formatted markdown section
                    f.write(temp_checker.format_for_markdown())

                    # Add explicit pointers to CSV artifacts for downstream analysis
                    csv_path = baseline_check_results.get('csv_path')
                    multivariate_csv_path = baseline_check_results.get('multivariate_csv_path')
                    if csv_path or multivariate_csv_path:
                        f.write("\n### Baseline Metrics CSV Exports\n\n")
                        if csv_path:
                            f.write(f"- **Univariate baseline metrics CSV:** `{csv_path}`\n")
                        if multivariate_csv_path:
                            f.write(f"- **Multivariate baseline metrics CSV:** `{multivariate_csv_path}`\n")
                        f.write(
                            "\nThese CSV files mirror the baseline diagnostics in a tabular format so you can "
                            "track learnability across runs, symbols, or execution modes.\n\n"
                        )

                    # Explain how to interpret Test R² and the quality score for learnability
                    f.write("### How to Interpret Baseline Learnability Metrics\n\n")
                    f.write(
                        "The baseline check fits simple models (linear regression and small LightGBM baselines) "
                        "on each feature individually, and on a few 2–3 feature combinations. This provides an "
                        "upper bound on how learnable the target is from the raw feature set alone, before "
                        "any complex modeling.\n\n"
                    )
                    f.write(
                        "- **Test R²** measures how much of the variance in the target is explained out-of-sample "
                        "by a given feature (or feature combination). Values near 0 mean the feature carries very "
                        "little predictive signal; values above roughly 0.3–0.4 indicate strong linear signal; "
                        "negative values indicate that even a simple model fails to generalize.\n"
                    )
                    f.write(
                        "- The **quality score** aggregates how many features achieve positive Test R², how strong "
                        "the best feature(s) are, and how consistent performance is across evaluated features. "
                        "Scores close to 1.0 mean that many features contain robust, learnable signal; scores near "
                        "0 indicate that almost all features behave like noise.\n\n"
                    )
                    f.write(
                        "In practice, a low quality score, many negative Test R² values, or a best feature with "
                        "weak Test R² suggests that labels/targets or feature definitions may need to be revisited "
                        "before investing further in complex downstream models.\n\n"
                    )

                f.write("## Next Steps\n\n")
                f.write("- Features are ready for feature selection and interaction generation\n")
                f.write("- Consider running lookback optimization for optimal feature parameters\n")
                f.write("- Proceed to labeling step for profit-target generation\n\n")
            
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate outcome report: {e}")
            return None

    def _calculate_comprehensive_feature_stats(self, features) -> Dict[str, Any]:
        """Calculate comprehensive statistics for all features."""
        try:
            import numpy as np
            import pandas as pd
            from scipy.stats import spearmanr, pearsonr
            from sklearn.feature_selection import mutual_info_regression
            
            stats = {
                'high_quality_count': 0,
                'medium_quality_count': 0,
                'low_quality_count': 0,
                'constant_features': 0,
                'high_correlation_pairs': 0,
                'avg_correlation': 0.0,
                'stability_score': 0.0,
                'top_features': [],
                'quality_distribution': {},
                'redundant_pairs': 0,
                'redundancy_rate': 0.0,
                'unique_features': 0,
                'redundancy_score': 0.0,
                'stable_features': 0,
                'moderate_stable': 0,
                'unstable_features': 0,
                'avg_stability': 0.0,
                'high_info_features': 0,
                'medium_info_features': 0,
                'low_info_features': 0,
                'avg_information': 0.0,
                'recommendations': {
                    'keep': [],
                    'remove': [],
                    'investigate': []
                }
            }
            
            if features is None or len(features.columns) == 0:
                return stats
            
            # Calculate feature quality scores
            feature_scores = {}
            feature_correlations = {}
            feature_stabilities = {}
            feature_information = {}
            
            # Get target column (returns or similar)
            target_col = None
            for col in ['returns', 'close_return', 'price_return', 'target']:
                if col in features.columns:
                    target_col = col
                    break
            
            if target_col is None:
                # Use first numeric column as proxy
                numeric_cols = features.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    target_col = numeric_cols[0]
            
            for col in features.columns:
                if col == target_col:
                    continue
                    
                try:
                    # Calculate correlation with target
                    valid_data = features[[col, target_col]].dropna()
                    if len(valid_data) < 10:
                        continue
                    
                    # Pearson correlation with robust error handling
                    try:
                        pearson_result = pearsonr(valid_data[col], valid_data[target_col])
                        if pearson_result and len(pearson_result) >= 1 and pearson_result[0] is not None:
                            pearson_corr = abs(float(pearson_result[0]))
                            if np.isnan(pearson_corr):
                                pearson_corr = 0.0
                        else:
                            pearson_corr = 0.0
                    except (ValueError, TypeError, IndexError):
                        pearson_corr = 0.0
                    
                    # Spearman correlation (rank correlation) with robust error handling
                    try:
                        spearman_result = spearmanr(valid_data[col], valid_data[target_col])
                        if spearman_result and len(spearman_result) >= 1 and spearman_result[0] is not None:
                            spearman_corr = abs(float(spearman_result[0]))
                            if np.isnan(spearman_corr):
                                spearman_corr = 0.0
                        else:
                            spearman_corr = 0.0
                    except (ValueError, TypeError, IndexError):
                        spearman_corr = 0.0
                    
                    # Feature stability (coefficient of variation)
                    feature_std = valid_data[col].std()
                    feature_mean = abs(valid_data[col].mean())
                    stability = 1 / (1 + feature_std / feature_mean) if feature_mean > 0 else 0.0
                    
                    # Information content (mutual information proxy)
                    try:
                        mi_score = mutual_info_regression(
                            valid_data[[col]], valid_data[target_col]
                        )[0]
                    except:
                        mi_score = 0.0
                    
                    # Combined quality score
                    quality_score = (
                        0.3 * pearson_corr + 
                        0.3 * spearman_corr + 
                        0.2 * stability + 
                        0.2 * mi_score
                    )
                    
                    feature_scores[col] = quality_score
                    feature_correlations[col] = max(pearson_corr, spearman_corr)
                    feature_stabilities[col] = stability
                    feature_information[col] = mi_score
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating stats for {col}: {e}")
                    continue
            
            # Count quality categories
            for score in feature_scores.values():
                if score > 0.7:
                    stats['high_quality_count'] += 1
                elif score > 0.4:
                    stats['medium_quality_count'] += 1
                else:
                    stats['low_quality_count'] += 1
            
            # Count constant features and collect their names
            constant_feature_names = []
            for col in features.columns:
                if col in feature_scores:
                    # More robust constant feature detection
                    col_data = features[col].dropna()  # Remove NaN values first
                    if len(col_data) == 0:
                        # All values are NaN, consider as constant
                        stats['constant_features'] += 1
                        constant_feature_names.append(col)
                    elif col_data.nunique() <= 1:
                        # Only 1 unique value (excluding NaN)
                        stats['constant_features'] += 1
                        constant_feature_names.append(col)
                    elif col_data.std() == 0:
                        # Zero standard deviation (all values identical)
                        stats['constant_features'] += 1
                        constant_feature_names.append(col)
            
            # Store constant feature names for reporting
            stats['constant_feature_names'] = constant_feature_names
            
            # Calculate correlation matrix for redundancy analysis
            numeric_features = features.select_dtypes(include=[np.number])
            if len(numeric_features.columns) > 1:
                corr_matrix = numeric_features.corr().abs()
                high_corr_pairs = (corr_matrix > 0.8).sum().sum() - len(corr_matrix.columns)
                stats['high_correlation_pairs'] = high_corr_pairs // 2  # Divide by 2 for symmetric matrix
                stats['avg_correlation'] = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            
            # Calculate stability metrics
            stability_scores = list(feature_stabilities.values())
            if stability_scores:
                stats['avg_stability'] = np.mean(stability_scores)
                for stability in stability_scores:
                    if stability > 0.8:
                        stats['stable_features'] += 1
                    elif stability > 0.5:
                        stats['moderate_stable'] += 1
                    else:
                        stats['unstable_features'] += 1
            
            # Calculate information content metrics
            info_scores = list(feature_information.values())
            if info_scores:
                stats['avg_information'] = np.mean(info_scores)
                for info in info_scores:
                    if info > 0.7:
                        stats['high_info_features'] += 1
                    elif info > 0.4:
                        stats['medium_info_features'] += 1
                    else:
                        stats['low_info_features'] += 1
            
            # Top features
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            stats['top_features'] = [
                (feature, {
                    'quality_score': score,
                    'correlation': feature_correlations.get(feature, 0.0),
                    'stability': feature_stabilities.get(feature, 0.0),
                    'information': feature_information.get(feature, 0.0)
                })
                for feature, score in sorted_features
            ]
            
            # Quality distribution
            if feature_scores:
                scores = list(feature_scores.values())
                stats['quality_distribution'] = {
                    'mean': np.mean(scores),
                    'median': np.median(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
            
            # Redundancy analysis
            if len(numeric_features.columns) > 1:
                corr_matrix = numeric_features.corr().abs()
                total_pairs = len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) // 2
                redundant_pairs = ((corr_matrix > 0.8).sum().sum() - len(corr_matrix.columns)) // 2
                stats['redundant_pairs'] = redundant_pairs
                stats['redundancy_rate'] = redundant_pairs / total_pairs if total_pairs > 0 else 0.0
                stats['unique_features'] = len(corr_matrix.columns) - redundant_pairs
                stats['redundancy_score'] = 1.0 - stats['redundancy_rate']
            
            # Overall stability score
            if feature_stabilities:
                stats['stability_score'] = np.mean(list(feature_stabilities.values()))
            
            # Generate recommendations
            for feature, score in sorted_features:
                if score > 0.7:
                    stats['recommendations']['keep'].append(feature)
                elif score < 0.4:
                    stats['recommendations']['remove'].append(feature)
                else:
                    stats['recommendations']['investigate'].append(feature)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive feature stats: {e}")
            return {
                'high_quality_count': 0,
                'medium_quality_count': 0,
                'low_quality_count': 0,
                'constant_features': 0,
                'high_correlation_pairs': 0,
                'avg_correlation': 0.0,
                'stability_score': 0.0,
                'top_features': [],
                'quality_distribution': {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                'redundant_pairs': 0,
                'redundancy_rate': 0.0,
                'unique_features': 0,
                'redundancy_score': 0.0,
                'stable_features': 0,
                'moderate_stable': 0,
                'unstable_features': 0,
                'avg_stability': 0.0,
                'high_info_features': 0,
                'medium_info_features': 0,
                'low_info_features': 0,
                'avg_information': 0.0,
                'recommendations': {'keep': [], 'remove': [], 'investigate': []}
            }

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Register the step
def register_feature_generation_feature_generation_step():
    """Register the feature generation step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("feature_generation_feature_generation_step", FeatureGenerationFeatureGenerationStep)
    tprint("✅ Feature generation step registered", "SUCCESS")


# Auto-register when module is imported
register_feature_generation_feature_generation_step()
