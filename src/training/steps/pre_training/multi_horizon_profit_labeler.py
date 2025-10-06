"""
Multi-Horizon Profit Labeler Component for Pre-Training Pipeline.

This component integrates the VolatilityAwareMultiHorizonLabeler with regime data splitting
to create differentiated profit labels for different market regimes.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.logger import system_logger

try:
    from src.utils.data.klines_parquet import get_klines_manager
except Exception:  # pragma: no cover - defensive guard for optional dependency
    get_klines_manager = None  # type: ignore[assignment]

# Import the volatility-aware multi-horizon labeler
from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
    VolatilityAwareMultiHorizonLabeler,
    VolatilityAwareConfig,
    LabelingResult
)

# Import base component
from src.training.steps.pre_training.components.base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult


@dataclass
class MultiHorizonConfig:
    """Configuration for multi-horizon profit labeling."""

    # Timeframe settings
    timeframe: str = "15m"
    base_period_minutes: float = 15.0

    # Volatility-aware labeling settings
    enable_volatility_normalization: bool = True
    enable_noise_gating: bool = True
    enable_quality_scoring: bool = True
    enable_multi_target_scheme: bool = True

    # Regime integration settings
    enable_regime_aware_labeling: bool = True
    regime_column: str = "regime_state"

    # Output settings
    min_data_points: int = 1000
    save_intermediate_results: bool = True
    generate_reports: bool = True

    # Quality thresholds
    min_auc_threshold: float = 0.55
    max_auc_std_threshold: float = 0.03
    min_psi_threshold: float = 0.1
    max_flip_rate_threshold: float = 0.15
    min_balance_threshold: float = 0.35
    max_balance_threshold: float = 0.65


class MultiHorizonProfitLabeler:
    """
    Multi-Horizon Profit Labeler that integrates volatility-aware labeling with regime data.

    This class creates differentiated profit labels for different market regimes,
    ensuring that the labeling process accounts for regime-specific behaviors.
    """

    def __init__(self, config: MultiHorizonConfig = None):
        """Initialize multi-horizon profit labeler."""
        self.config = config or MultiHorizonConfig()
        self.logger = logging.getLogger('MultiHorizonProfitLabeler')

        # Initialize the volatility-aware labeler
        self.volatility_labeler = VolatilityAwareMultiHorizonLabeler(self._create_volatility_config())

        tprint_success("🚀 Multi-Horizon Profit Labeler initialized")
        tprint_info(f"   → Timeframe: {self.config.timeframe}")
        tprint_info(f"   → Regime-aware: {self.config.enable_regime_aware_labeling}")
        tprint_info(f"   → Volatility normalization: {self.config.enable_volatility_normalization}")

    def _create_volatility_config(self) -> VolatilityAwareConfig:
        """Create volatility-aware configuration from multi-horizon config."""
        return VolatilityAwareConfig(
            min_data_points=self.config.min_data_points,
            generate_reports=self.config.generate_reports,
            save_intermediate_results=self.config.save_intermediate_results,
            min_auc_threshold=self.config.min_auc_threshold,
            max_auc_std_threshold=self.config.max_auc_std_threshold
        )

    async def execute_labeling(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = "historical_data",
        regime_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute multi-horizon profit labeling.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'binance')
            timeframe: Timeframe for labeling (e.g., '15m')
            data_dir: Directory containing historical data
            regime_data: Optional regime data for regime-aware labeling

        Returns:
            Dictionary containing labeling results and metadata
        """
        try:
            tprint_info(f"🏷️ Starting multi-horizon profit labeling for {symbol} on {exchange}")
            tprint_info(f"⏰ Timeframe: {timeframe}")

            # Load market data
            market_data = await self._load_market_data(symbol, exchange, timeframe, data_dir)
            if market_data is None or market_data.empty:
                raise ValueError(f"No market data available for {symbol} {timeframe}")

            # Apply regime-aware labeling if enabled and regime data is available
            if self.config.enable_regime_aware_labeling and regime_data:
                labeling_result = await self._execute_regime_aware_labeling(market_data, regime_data)
            else:
                # Use standard volatility-aware labeling
                labeling_result = self.volatility_labeler.generate_labels(market_data)

            # Generate comprehensive report using the profit labeling report generator
            report = await self._generate_comprehensive_report(
                labeling_result, symbol, exchange, timeframe, regime_data
            )

            # Map target columns to expected names for feature lookback optimization compatibility
            mapped_labels = self._map_target_columns_for_feature_optimization(labeling_result.labels)

            # Create properly structured artifacts that feature lookback optimization expects
            # The feature lookback optimization expects 'labeled_data' or 'labels' keys
            artifacts = {
                'multi_horizon_labeling_result': {
                    'labeled_data': mapped_labels,  # This is what feature lookback optimization expects
                    'labels': mapped_labels,  # Backward compatibility
                    'confidence_scores': labeling_result.confidence_scores,
                    'eligibility_masks': labeling_result.eligibility_masks,
                    'quality_scores': labeling_result.quality_scores,
                    'method': 'multi_horizon_profit_labeling',
                    'metadata': {
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'regime_aware': self.config.enable_regime_aware_labeling and regime_data is not None,
                        'processing_time': labeling_result.processing_time,
                        'n_samples': labeling_result.n_samples,
                        'n_targets': labeling_result.n_targets,
                        'n_horizons': labeling_result.n_horizons
                    }
                },
                'labeling_report': report
            }

            tprint_success(f"✅ Multi-horizon labeling completed for {symbol}")
            tprint_info(f"   → Samples: {labeling_result.n_samples}")
            tprint_info(f"   → Targets: {labeling_result.n_targets}")
            tprint_info(f"   → Processing time: {labeling_result.processing_time:.2f}s")

            return artifacts

        except Exception as e:
            tprint_error(f"❌ Multi-horizon labeling failed: {e}")
            return {
                'multi_horizon_labeling_result': {},
                'labeling_report': {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }

    async def _load_market_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> Optional[pd.DataFrame]:
        """Load market data for the specified symbol and timeframe."""

        if get_klines_manager is None:
            message = (
                "kline_parquet utilities are not available. "
                "Ensure src.utils.data.klines_parquet can be imported."
            )
            self.logger.error(message)
            tprint_error(f"❌ {message}")
            raise RuntimeError(message)

        tprint_info(f"📊 Loading market data for {symbol} {timeframe} from {data_dir}")

        manager = get_klines_manager(data_dir)

        symbol_variants = list(dict.fromkeys([symbol, symbol.upper(), symbol.lower()]))
        timeframe_variants = list(dict.fromkeys([timeframe, timeframe.lower(), timeframe.upper()]))
        data_type_variants = ["processed", "raw"]

        load_errors: List[str] = []

        for sym in symbol_variants:
            for tf in timeframe_variants:
                for data_type in data_type_variants:
                    try:
                        tprint_info(
                            f"🔍 Attempting klines_parquet load for {sym}/{tf} [{data_type}]"
                        )
                        raw_df = await asyncio.to_thread(
                            manager.read_data,
                            sym,
                            tf,
                            None,
                            None,
                            data_type,
                        )
                    except Exception as load_error:  # pragma: no cover - defensive guard
                        error_msg = (
                            f"Failed to load {sym}/{tf} ({data_type}) via klines_parquet: {load_error}"
                        )
                        self.logger.warning(error_msg)
                        load_errors.append(error_msg)
                        continue

                    if raw_df is None or raw_df.empty:
                        info_msg = (
                            f"klines_parquet returned no data for {sym}/{tf} ({data_type})"
                        )
                        self.logger.info(info_msg)
                        load_errors.append(info_msg)
                        continue

                    try:
                        prepared = self._prepare_market_data_frame(raw_df)
                    except Exception as prep_error:
                        prep_msg = (
                            f"Loaded data for {sym}/{tf} ({data_type}) could not be prepared: {prep_error}"
                        )
                        self.logger.warning(prep_msg)
                        load_errors.append(prep_msg)
                        continue

                    tprint_success(
                        f"✅ Loaded {len(prepared)} rows via klines_parquet for {sym} {tf}"
                    )
                    return prepared

        error_message = (
            f"No market data available for {symbol} on {exchange} with timeframe {timeframe}."
        )
        if load_errors:
            for msg in load_errors[-5:]:  # Log the most recent errors for context
                self.logger.error(msg)
        self.logger.error(error_message)
        tprint_error(f"❌ {error_message}")
        raise FileNotFoundError(error_message)

    def _prepare_market_data_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure loaded market data is indexed and typed as expected by the labeler."""
        if data is None or data.empty:
            raise ValueError("Loaded market data is empty")

        df = data.copy()

        if "timestamp" in df.columns:
            timestamp_series = df.pop("timestamp")
        elif "open_time" in df.columns:
            timestamp_series = df.pop("open_time")
        elif df.index.name == "timestamp":
            timestamp_series = df.index
        else:
            timestamp_series = df.index

        ts = pd.to_datetime(timestamp_series, utc=True, errors="coerce")
        if ts.isnull().any():
            # Try integer timestamps (milliseconds/seconds)
            numeric_ts = pd.to_numeric(timestamp_series, errors="coerce")
            if numeric_ts.notnull().all():
                unit = "ms" if numeric_ts.max() > 10**12 else "s"
                ts = pd.to_datetime(numeric_ts, unit=unit, utc=True, errors="coerce")
        if ts.isnull().all():
            raise ValueError("Unable to parse timestamps for market data")

        ts_index = pd.DatetimeIndex(ts)

        valid_mask = ~pd.isna(ts_index)
        if not valid_mask.all():
            df = df.loc[valid_mask]
            ts_index = ts_index[valid_mask]
        if ts_index.empty:
            raise ValueError("Market data contains no valid timestamps")

        if ts_index.tz is not None:
            ts_index = ts_index.tz_convert(None)
        else:
            ts_index = ts_index.tz_localize(None)

        df.index = ts_index

        # Normalize column names
        normalized_columns = {col: col.lower() for col in df.columns}
        df = df.rename(columns=normalized_columns)

        volume_candidates = [
            "volume",
            "volume_usdt",
            "quote_volume",
            "vol",
        ]
        if "volume" not in df.columns:
            for candidate in volume_candidates:
                if candidate in df.columns:
                    df["volume"] = df.pop(candidate)
                    break

        required_columns = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Market data missing required columns: {missing}")

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required_columns)
        if df.empty:
            raise ValueError("Market data contains no valid OHLCV rows after cleaning")

        return df

    async def _execute_regime_aware_labeling(self, market_data: pd.DataFrame, regime_data: Dict[str, Any]) -> LabelingResult:
        """
        Execute regime-aware labeling that creates differentiated labels for different regimes.

        Args:
            market_data: Market data
            regime_data: Regime data from regime_data_splitting

        Returns:
            LabelingResult with regime-differentiated labels
        """
        try:
            tprint_info("🎭 Executing regime-aware labeling")

            # Extract regime assignments from regime data
            regime_assignments = self._extract_regime_assignments(market_data, regime_data)
            if regime_assignments is None:
                tprint_warning("⚠️ No regime assignments found, falling back to standard labeling")
                return self.volatility_labeler.generate_labels(market_data)

            # Get unique regimes
            regimes = np.unique(regime_assignments[~pd.isna(regime_assignments)])
            tprint_info(f"📊 Found {len(regimes)} distinct regimes")

            if len(regimes) == 0:
                tprint_warning("⚠️ No valid regime assignments, falling back to standard labeling")
                return self.volatility_labeler.generate_labels(market_data)

            # Create regime-specific labels using the volatility-aware labeler for each regime
            regime_labels = {}
            regime_quality_scores = {}
            total_processing_time = 0.0

            for regime in regimes:
                tprint_info(f"🏷️ Processing regime {regime}")

                # Filter data for this regime
                regime_mask = regime_assignments == regime
                regime_data_subset = market_data[regime_mask].copy()

                if len(regime_data_subset) < self.config.min_data_points:
                    tprint_warning(f"⚠️ Insufficient data for regime {regime}: {len(regime_data_subset)} samples")
                    continue

                # Generate labels for this regime using the volatility-aware labeler
                regime_result = self.volatility_labeler.generate_labels(regime_data_subset)

                if not regime_result.labels.empty:
                    # Add regime suffix to column names to differentiate between regimes
                    regime_labels[regime] = regime_result.labels.add_suffix(f'_regime_{regime}')
                    regime_quality_scores.update({
                        f"{target}_regime_{regime}": quality_score
                        for target, quality_score in regime_result.quality_scores.items()
                    })
                    total_processing_time += regime_result.processing_time

            # Combine regime-specific labels
            if regime_labels:
                combined_labels = pd.concat(regime_labels.values(), axis=1)

                # Create combined result with proper metadata
                combined_result = LabelingResult(
                    labels=combined_labels,
                    confidence_scores=pd.DataFrame(index=combined_labels.index),  # Will be populated by individual regime results
                    eligibility_masks=pd.DataFrame(index=combined_labels.index),   # Will be populated by individual regime results
                    quality_scores=regime_quality_scores,
                    n_samples=len(combined_labels),
                    n_targets=len([col for col in combined_labels.columns if 'target' in col]),
                    processing_time=total_processing_time
                )

                tprint_success(f"✅ Regime-aware labeling completed for {len(regime_labels)} regimes")
                return combined_result
            else:
                tprint_warning("⚠️ No valid regime-specific labels generated, falling back to standard labeling")
                return self.volatility_labeler.generate_labels(market_data)

        except Exception as e:
            tprint_error(f"❌ Regime-aware labeling failed: {e}")
            # Fall back to standard labeling
            return self.volatility_labeler.generate_labels(market_data)

    def _extract_regime_assignments(self, market_data: pd.DataFrame, regime_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract regime assignments from regime data.

        Args:
            market_data: Market data
            regime_data: Regime data from regime_data_splitting

        Returns:
            Array of regime assignments or None if not found
        """
        try:
            # Try to get regime assignments from regime data
            if 'regime_data' in regime_data:
                regime_info = regime_data['regime_data']

                # Check if regime states are directly available
                if 'regime_states' in regime_info:
                    regime_states = regime_info['regime_states']
                    if len(regime_states) == len(market_data):
                        return regime_states

                # Check if market data in regime data has regime column
                if 'market_data' in regime_info and regime_info['market_data'] is not None:
                    regime_market_data = regime_info['market_data']
                    if self.config.regime_column in regime_market_data.columns:
                        return regime_market_data[self.config.regime_column].values

            # Check if regime assignments are in the market data itself
            if self.config.regime_column in market_data.columns:
                return market_data[self.config.regime_column].values

            tprint_warning(f"⚠️ No regime assignments found in regime data or market data")
            return None

        except Exception as e:
            tprint_warning(f"⚠️ Error extracting regime assignments: {e}")
            return None

    def _map_target_columns_for_feature_optimization(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map target column names to expected names for feature lookback optimization compatibility.

        Feature lookback optimization expects specific target column names like:
        - 'leverage_adjusted_score'
        - 'immediate_opportunity'
        - 'short_term_opportunity'

        This method maps the generated target columns (like 'small_k0.50_a1.00') to these expected names.
        """
        try:
            if labels_df is None or labels_df.empty:
                tprint_warning("⚠️ No labels to map for feature optimization compatibility")
                return labels_df

            mapped_df = labels_df.copy()
            tprint_info(f"🔄 Mapping {len(labels_df.columns)} target columns for feature optimization compatibility")

            # Define mapping from generated column patterns to expected names
            column_mappings = {
                'leverage_adjusted_score': [],
                'immediate_opportunity': [],
                'short_term_opportunity': []
            }

            # Priority 1: Map small band targets to immediate_opportunity (shortest horizon)
            # Handle both regular targets and regime-specific targets
            small_targets = [col for col in labels_df.columns if 'small_' in col and '_regime_' not in col]
            small_regime_targets = [col for col in labels_df.columns if 'small_' in col and '_regime_' in col]

            # Use regular targets first, then regime targets if no regular targets available
            if small_targets:
                best_small_target = self._select_best_target_by_pattern(small_targets, labels_df, 'small')
                if best_small_target:
                    column_mappings['immediate_opportunity'].append(best_small_target)
            elif small_regime_targets:
                # Use the first regime target (could be improved to select best regime)
                best_small_target = self._select_best_target_by_pattern(small_regime_targets, labels_df, 'small')
                if best_small_target:
                    column_mappings['immediate_opportunity'].append(best_small_target)

            # Priority 2: Map medium band targets to short_term_opportunity (medium horizon)
            medium_targets = [col for col in labels_df.columns if 'medium_' in col and '_regime_' not in col]
            medium_regime_targets = [col for col in labels_df.columns if 'medium_' in col and '_regime_' in col]

            if medium_targets:
                best_medium_target = self._select_best_target_by_pattern(medium_targets, labels_df, 'medium')
                if best_medium_target:
                    column_mappings['short_term_opportunity'].append(best_medium_target)
            elif medium_regime_targets:
                best_medium_target = self._select_best_target_by_pattern(medium_regime_targets, labels_df, 'medium')
                if best_medium_target:
                    column_mappings['short_term_opportunity'].append(best_medium_target)

            # Priority 3: Map high band targets to leverage_adjusted_score (longest horizon)
            high_targets = [col for col in labels_df.columns if 'high_' in col and '_regime_' not in col]
            high_regime_targets = [col for col in labels_df.columns if 'high_' in col and '_regime_' in col]

            if high_targets:
                best_high_target = self._select_best_target_by_pattern(high_targets, labels_df, 'high')
                if best_high_target:
                    column_mappings['leverage_adjusted_score'].append(best_high_target)
            elif high_regime_targets:
                best_high_target = self._select_best_target_by_pattern(high_regime_targets, labels_df, 'high')
                if best_high_target:
                    column_mappings['leverage_adjusted_score'].append(best_high_target)

            # Apply the mappings
            for expected_name, source_columns in column_mappings.items():
                if source_columns:
                    # Use the first (best) source column
                    source_col = source_columns[0]
                    if source_col in mapped_df.columns:
                        mapped_df[expected_name] = mapped_df[source_col]
                        tprint_info(f"✅ Mapped '{source_col}' → '{expected_name}'")

            # Also add the original columns for backward compatibility and debugging
            tprint_info(f"✅ Target column mapping completed. Original: {len(labels_df.columns)}, Mapped: {len(mapped_df.columns)}")

            return mapped_df

        except Exception as e:
            tprint_warning(f"⚠️ Error mapping target columns: {e}")
            # Return original dataframe if mapping fails
            return labels_df

    def _select_best_target_by_pattern(self, target_columns: List[str], labels_df: pd.DataFrame, pattern: str) -> Optional[str]:
        """
        Select the best target column from a list of candidates based on pattern and data quality.

        Args:
            target_columns: List of column names matching the pattern
            labels_df: DataFrame with the labels
            pattern: Pattern type ('small', 'medium', 'high')

        Returns:
            Best column name or None if no suitable column found
        """
        try:
            if not target_columns:
                return None

            # For now, select the first target in the list
            # In a more sophisticated implementation, we could analyze label quality,
            # balance, predictability, etc. to select the best target
            selected_target = target_columns[0]

            # Validate that the selected target has reasonable data
            if selected_target in labels_df.columns:
                target_data = labels_df[selected_target].dropna()

                # Check if we have enough non-null values
                if len(target_data) > 100:  # Minimum threshold for reliable analysis
                    tprint_info(f"✅ Selected '{selected_target}' as best {pattern} target")
                    return selected_target

            tprint_warning(f"⚠️ No suitable {pattern} target found among {len(target_columns)} candidates")
            return None

        except Exception as e:
            tprint_warning(f"⚠️ Error selecting best target for pattern {pattern}: {e}")
            return target_columns[0] if target_columns else None

    async def _generate_comprehensive_report(
        self,
        labeling_result: LabelingResult,
        symbol: str,
        exchange: str,
        timeframe: str,
        regime_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive labeling report with regime-aware analysis."""
        try:
            # Import the profit labeling report generator
            from src.training.steps.pre_training.profit_labeling.profit_labeling_report_generator import (
                ProfitLabelingReportGenerator, ProfitLabelingReport
            )

            tprint_info("📋 Generating comprehensive profit labeling report")

            # Create the report generator
            report_generator = ProfitLabelingReportGenerator()

            # Prepare the labeling result data for the report generator
            labeling_result_data = {
                'multi_horizon_labeling_result': {
                    'labeled_data': labeling_result.labels,
                    'confidence_scores': labeling_result.confidence_scores,
                    'eligibility_masks': labeling_result.eligibility_masks,
                    'quality_scores': labeling_result.quality_scores,
                    'metadata': {
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'regime_aware': self.config.enable_regime_aware_labeling and regime_data is not None,
                        'processing_time': labeling_result.processing_time,
                        'n_samples': labeling_result.n_samples,
                        'n_targets': labeling_result.n_targets,
                        'n_horizons': labeling_result.n_horizons
                    }
                },
                'labeling_report': self._generate_basic_labeling_report(labeling_result, symbol, exchange, timeframe)
            }

            # Generate the comprehensive report
            comprehensive_report = report_generator.generate_report(
                labeling_result=labeling_result_data,
                regime_data=regime_data,
                output_directory="profit_labeling_reports"
            )

            # Convert the report object to dictionary for pipeline compatibility
            report_dict = {
                'status': 'completed',
                'symbol': comprehensive_report.symbol,
                'exchange': comprehensive_report.exchange,
                'timeframe': comprehensive_report.timeframe,
                'timestamp': comprehensive_report.timestamp.isoformat(),
                'processing_time': comprehensive_report.processing_time,
                'statistics': {
                    'n_samples': comprehensive_report.n_samples,
                    'n_targets': comprehensive_report.n_targets,
                    'n_horizons': comprehensive_report.n_horizons,
                    'label_distribution': comprehensive_report.label_distribution
                },
                'quality_scores': comprehensive_report.quality_scores,
                'regime_statistics': comprehensive_report.regime_statistics,
                'feature_lookback_compatibility': comprehensive_report.feature_lookback_compatibility,
                'recommendations': comprehensive_report.recommendations
            }

            tprint_success("✅ Comprehensive profit labeling report generated")
            return report_dict

        except Exception as e:
            tprint_warning(f"⚠️ Error generating comprehensive report: {e}")
            # Fall back to basic report generation
            return self._generate_basic_labeling_report(labeling_result, symbol, exchange, timeframe)

    def _generate_basic_labeling_report(
        self,
        labeling_result: LabelingResult,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Generate basic labeling report as fallback."""
        try:
            report = {
                'status': 'completed',
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'processing_time': labeling_result.processing_time,
                'statistics': {
                    'n_samples': labeling_result.n_samples,
                    'n_targets': labeling_result.n_targets,
                    'n_horizons': labeling_result.n_horizons,
                    'label_distribution': labeling_result.label_distribution
                },
                'quality_summary': {}
            }

            # Add quality scores summary
            if labeling_result.quality_scores:
                quality_summary = {}
                for target_name, quality_score in labeling_result.quality_scores.items():
                    quality_summary[target_name] = {
                        'overall_quality': quality_score.overall_quality,
                        'predictability': quality_score.predictability,
                        'stability': quality_score.stability,
                        'balance': quality_score.balance,
                        'auc_mean': quality_score.auc_mean,
                        'class_balance': quality_score.class_balance
                    }
                report['quality_summary'] = quality_summary

            return report

        except Exception as e:
            tprint_warning(f"⚠️ Error generating basic report: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class MultiHorizonProfitLabelerComponent(BasePreTrainingComponent):
    """
    Component wrapper for Multi-Horizon Profit Labeler.

    This component integrates with the pre-training pipeline and handles
    regime-aware profit labeling with proper error handling and reporting.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the multi-horizon profit labeler component."""
        super().__init__(config)
        self.labeler = None

        # Create configuration from component config
        mh_config = MultiHorizonConfig()

        # Override with custom parameters if provided
        if config and config.custom_params:
            for key, value in config.custom_params.items():
                if hasattr(mh_config, key):
                    setattr(mh_config, key, value)

        self.labeler = MultiHorizonProfitLabeler(mh_config)

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['multi_horizon_labeling_result', 'labeling_report']

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute multi-horizon profit labeling as a component.

        Args:
            data: Input data (typically None for this component)
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with labeling results
        """
        try:
            # Extract parameters from pipeline state
            symbol = pipeline_state.get('symbol', 'ETHUSDT')
            exchange = pipeline_state.get('exchange', 'binance')
            timeframe = pipeline_state.get('timeframe', '15m')
            data_dir = pipeline_state.get('data_dir', 'historical_data')

            # Extract regime data from pipeline state if available
            regime_data = pipeline_state.get('regime_data_splitting_result')

            # Execute labeling with regime data if available
            labeling_result = await self.labeler.execute_labeling(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                regime_data=regime_data
            )

            # Save artifacts persistently for other components to use
            try:
                # Save the complete artifacts structure as a single outcome file
                # that the feature lookback optimization can load
                outcome_data = {
                    'config': {
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe
                    },
                    'artifacts': labeling_result,
                    'metadata': {
                        'component_type': 'multi_horizon_profit_labeler',
                        'saved_at': datetime.now().isoformat()
                    }
                }

                # Save as a single outcome file that matches the expected pattern
                import json
                outcomes_dir = Path("outcomes")
                outcomes_dir.mkdir(exist_ok=True)

                filename = f"market_analysis_multi_horizon_profit_labeler_outcome_{symbol}_{exchange}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                outcome_file = outcomes_dir / filename

                with open(outcome_file, 'w') as f:
                    json.dump(outcome_data, f, indent=2, default=str)

                tprint_info(f"💾 Labeling outcome saved to {outcome_file}")

            except Exception as e:
                tprint_warning(f"⚠️ Failed to save outcome: {e}")

            return ComponentResult(
                success=True,
                artifacts=labeling_result,
                metadata={
                    'component_type': 'multi_horizon_profit_labeler',
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'artifacts_saved': True
                }
            )

        except Exception as e:
            tprint_error(f"❌ Multi-horizon profit labeler component failed: {e}")
            return ComponentResult(
                success=False,
                artifacts={
                    'multi_horizon_labeling_result': {},
                    'labeling_report': {
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                },
                error_message=str(e),
                metadata={'component_type': 'multi_horizon_profit_labeler'}
            )