"""
Feature Generation Interaction Generation Step

This step generates feature interactions via the consolidated pipeline runner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from tprint import tprint

from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import (
    run_interaction_generation_step
)
from src.training.steps.pre_training.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent
)
from src.training.steps.pre_training.utils.artifact_manager import (
    get_pretraining_artifact_manager,
    ArtifactKeys,
)

# Import CMI complementarity components
try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_complementarity import (
        CMIComplementarityScorer, CMIComplementarityConfig
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.analyst_side_info import (
        AnalystSideInfoHandler, AnalystSideInfoConfig
    )
    CMI_COMPLEMENTARITY_AVAILABLE = True
except ImportError:
    CMI_COMPLEMENTARITY_AVAILABLE = False
    CMIComplementarityScorer = None
    CMIComplementarityConfig = None
    AnalystSideInfoHandler = None
    AnalystSideInfoConfig = None


@dataclass
class InteractionGenerationResult:
    success: bool
    interaction_features: pd.DataFrame
    interaction_metadata: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class FeatureGenerationInteractionGenerationStep(ModularComponent):
    """Interaction generation step that calls the consolidated pipeline."""

    def __init__(self, name: str = "interaction_generation_step",
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(name, config or {}, logger)
        
        # Initialize CMI complementarity components if available
        if CMI_COMPLEMENTARITY_AVAILABLE:
            # CMI configuration for interaction generation
            cmi_config = CMIComplementarityConfig(
                per_family_budget=(3, 8),  # Fewer interactions per family
                upstream_multiplier=2,  # Total budget to RFE = 2× per-family
                max_total_features=30,  # Maximum total interactions to select
                enable_regime_awareness=True,  # Compute R(X|A) per regime
                compute_timeout_seconds=300.0,  # 5 min hard limit
                interaction_gain_percentile=75  # Accept if > 75th percentile of null
            )
            self.cmi_scorer = CMIComplementarityScorer(cmi_config)
            self.analyst_handler = AnalystSideInfoHandler()
        else:
            self.cmi_scorer = None
            self.analyst_handler = None

    async def execute(self,
                      training_input: Dict[str, Any],
                      pipeline_state: Dict[str, Any]) -> InteractionGenerationResult:
        tprint("🔧 Starting interaction generation via consolidated pipeline")
        self.logger.info("🔧 Starting interaction generation via consolidated pipeline")
        
        # Check if CMI complementarity is enabled (Tactician mode only)
        enable_cmi_complementarity = (
            CMI_COMPLEMENTARITY_AVAILABLE and 
            self.cmi_scorer is not None and 
            pipeline_state is not None and 
            pipeline_state.get('tactician_mode', False)
        )
        
        tprint(f"🎯 CMI complementarity check: CMI_AVAILABLE={CMI_COMPLEMENTARITY_AVAILABLE}, cmi_scorer={self.cmi_scorer is not None}, tactician_mode={pipeline_state.get('tactician_mode', False) if pipeline_state else None}")
        
        if enable_cmi_complementarity:
            tprint("🎯 CMI complementarity enabled for Tactician mode interaction generation")
            self.logger.info("🎯 CMI complementarity enabled for Tactician mode interaction generation")
        else:
            tprint("📊 Standard interaction generation (Analyst mode or CMI unavailable)")
            self.logger.info("📊 Standard interaction generation (Analyst mode or CMI unavailable)")

        # Get artifact manager
        tprint("📦 Getting artifact manager")
        artifact_manager = get_pretraining_artifact_manager()
        
        # Try to load from artifact manager first
        tprint("🔍 Checking for cached interaction features")
        cached_interactions = artifact_manager.retrieve_enhanced(ArtifactKeys.INTERACTION_FEATURES)
        cached_metadata = artifact_manager.retrieve_enhanced(ArtifactKeys.INTERACTION_METADATA)
        cached_metrics = artifact_manager.retrieve_enhanced(ArtifactKeys.INTERACTION_GENERATION_METRICS)
        
        tprint(f"📦 Cache check: interactions={cached_interactions is not None}, metadata={cached_metadata is not None}, metrics={cached_metrics is not None}")
        
        if cached_interactions is not None:
            tprint("📦 Retrieved interaction features from artifact manager")
            self.logger.info("📦 Retrieved interaction features from artifact manager")
            result_cached = InteractionGenerationResult(
                success=True,
                interaction_features=cached_interactions,
                interaction_metadata=cached_metadata or {},
                generation_metrics=cached_metrics or {},
                artifacts={'cache_hit': True},
                error_message=None
            )
            # Best-effort report from cache
            tprint("📊 Generating report from cached data")
            try:
                symbol = training_input.get('symbol', 'ETHUSDT')
                timeframe = training_input.get('timeframe', '15m')
                data_for_metrics = training_input.get('data')
                tprint(f"📊 Report params: symbol={symbol}, timeframe={timeframe}, data_available={data_for_metrics is not None}")
                report = self._generate_interaction_report(
                    result_cached.interaction_features,
                    result_cached.interaction_metadata,
                    symbol,
                    timeframe,
                    data_for_metrics
                )
                md = self._format_interaction_markdown(report)
                self._store_interaction_report(report, md, symbol, timeframe)
                tprint("📊 Report generated and stored successfully")
            except Exception as e:
                tprint(f"⚠️ Report generation failed: {e}")
                pass
            tprint("✅ Returning cached result")
            return result_cached

        tprint("🔍 Extracting training input parameters")
        data = training_input.get('data')
        symbol = training_input.get('symbol', 'ETHUSDT')
        timeframe = training_input.get('timeframe', '15m')
        direction = training_input.get('direction', 'longs')
        intensity = training_input.get('intensity', 'blank')
        lookback_days = training_input.get('lookback_days')
        start_date = training_input.get('start_date')
        end_date = training_input.get('end_date')
        exchange = training_input.get('exchange', 'binance')
        custom_overrides = training_input.get('custom_overrides')
        
        tprint(f"📊 Input params: symbol={symbol}, timeframe={timeframe}, direction={direction}, intensity={intensity}")
        tprint(f"📊 Data params: data_shape={data.shape if hasattr(data, 'shape') else 'None'}, lookback_days={lookback_days}, start_date={start_date}, end_date={end_date}")
        tprint(f"📊 Exchange: {exchange}, custom_overrides={custom_overrides is not None}")

        # Enforce using only selected features from feature selection step
        try:
            tprint("🔍 Loading selected features from artifact manager")
            selected_df = artifact_manager.get_dataframe('feature_selection', ArtifactKeys.SELECTED_FEATURES)
            if (selected_df is None or selected_df.empty):
                # Backward-compatibility: alternative step naming
                selected_df = artifact_manager.get_dataframe('feature_generation_feature_selection_step', ArtifactKeys.SELECTED_FEATURES)
            if selected_df is None or selected_df.empty:
                tprint("❌ No selected features available; interaction generation requires prior feature selection")
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={},
                    generation_metrics={},
                    artifacts={},
                    error_message="Selected features not found. Run feature_selection before interaction_generation."
                )
            tprint(f"✅ Using selected features for interaction generation: shape={selected_df.shape}")
            data = selected_df

            # Load labeling targets and align with selected features
            targets_series: Optional[pd.Series] = None
            for step_name in ("feature_generation_labeling_integration_step", "labeling_integration"):
                series = artifact_manager.get_series(step_name, ArtifactKeys.TARGETS)
                if isinstance(series, pd.Series) and not series.empty:
                    targets_series = series.astype(float)
                    tprint(f"✅ Loaded labeling targets from {step_name}: count={len(targets_series)}")
                    break

            if targets_series is None or targets_series.empty:
                tprint("❌ Labeling targets not found for interaction generation")
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={},
                    generation_metrics={},
                    artifacts={},
                    error_message="Targets from feature_generation_labeling_integration_step are required before interaction generation."
                )

            aligned = data.join(targets_series.rename("target"), how="inner").dropna(axis=0, how="any")
            if aligned.empty:
                tprint("❌ No overlapping timestamps between selected features and labeling targets")
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={},
                    generation_metrics={},
                    artifacts={},
                    error_message="No overlapping timestamps between selected features and labeling targets."
                )

            targets = aligned.pop("target")
            data = aligned
            tprint(f"✅ Aligned features/targets for interaction generation: features={data.shape}, targets={targets.shape}")
        except Exception as e:
            return InteractionGenerationResult(
                success=False,
                interaction_features=pd.DataFrame(),
                interaction_metadata={},
                generation_metrics={},
                artifacts={},
                error_message=f"Failed to load selected features: {e}"
            )

        try:
            # Load optimized periods/lookbacks if available and pass in overrides
            try:
                opt_periods = artifact_manager.get_artifact('period_lookback_optimization', 'optimized_periods')
                opt_lookbacks = artifact_manager.get_artifact('period_lookback_optimization', 'optimized_lookbacks')
            except Exception:
                opt_periods, opt_lookbacks = None, None
            if custom_overrides is None:
                custom_overrides = {}
            if not isinstance(custom_overrides, dict):
                custom_overrides = dict(custom_overrides)
            if isinstance(custom_overrides, dict):
                if opt_periods is not None:
                    custom_overrides.setdefault('optimized_periods', opt_periods)
                if opt_lookbacks is not None:
                    custom_overrides.setdefault('optimized_lookbacks', opt_lookbacks)
            custom_overrides.setdefault('targets', targets)
            pipeline_state = dict(pipeline_state)
            pipeline_state['targets'] = targets

            tprint("🚀 Calling run_interaction_generation_step")
            result = await run_interaction_generation_step(
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                intensity=intensity,
                lookback_days=lookback_days,
                start_date=start_date,
                end_date=end_date,
                exchange=exchange,
                custom_overrides=custom_overrides
            )
            tprint(f"✅ run_interaction_generation_step completed: success={result.get('success', False)}")

            # Store artifacts in artifact manager
            if result.get('success', False):
                tprint("📦 Processing successful result")
                interaction_features = result.get('interaction_features', pd.DataFrame())
                interaction_metadata = result.get('interaction_metadata', {})
                generation_metrics = result.get('generation_metrics', {})
                
                tprint(f"📊 Result data: features_shape={interaction_features.shape if hasattr(interaction_features, 'shape') else 'None'}, metadata_keys={list(interaction_metadata.keys()) if interaction_metadata else []}")
                
                # Attach optimized period/lookback info if available
                try:
                    opt_periods = artifact_manager.get_artifact('period_lookback_optimization', 'optimized_periods')
                    opt_lookbacks = artifact_manager.get_artifact('period_lookback_optimization', 'optimized_lookbacks')
                    if isinstance(interaction_metadata, dict):
                        if opt_periods is not None:
                            interaction_metadata.setdefault('optimized_periods', opt_periods)
                        if opt_lookbacks is not None:
                            interaction_metadata.setdefault('optimized_lookbacks', opt_lookbacks)
                except Exception:
                    pass

                # Apply CMI complementarity filtering if enabled
                if enable_cmi_complementarity and not interaction_features.empty:
                    tprint("🎯 Applying CMI complementarity filtering to interaction features")
                    self.logger.info("🎯 Applying CMI complementarity filtering to interaction features")
                    try:
                        # Get targets from pipeline state
                        targets = pipeline_state.get('targets')
                        tprint(f"🎯 CMI targets check: targets_available={targets is not None}")
                        if targets is not None:
                            tprint("🎯 Extracting Analyst side information")
                            # Extract Analyst side information
                            analyst_result = self.analyst_handler.extract_side_info(
                                pipeline_state, targets, interaction_features.index
                            )
                            tprint(f"🎯 Analyst result: is_valid={analyst_result.is_valid}, degraded={analyst_result.degraded_to_unconditional}, source={analyst_result.source}")
                            
                            if analyst_result.is_valid and not analyst_result.degraded_to_unconditional:
                                tprint("🎯 Applying CMI complementarity scoring for interactions")
                                # Apply CMI complementarity scoring for interactions
                                # Use conditional gain: I(Y; fi∘fj | A, fi, fj) > threshold
                                cmi_result = self.cmi_scorer.score_features(
                                    interaction_features, targets, analyst_result.A,
                                    pipeline_state=pipeline_state
                                )
                                tprint(f"🎯 CMI scoring result: is_valid={cmi_result.is_valid}, selected_count={len(cmi_result.selected_features) if cmi_result.selected_features else 0}")
                                
                                if cmi_result.is_valid and cmi_result.selected_features:
                                    # Filter interactions based on CMI selection
                                    original_count = len(interaction_features.columns)
                                    interaction_features = interaction_features[cmi_result.selected_features]
                                    filtered_count = len(interaction_features.columns)
                                    
                                    tprint(f"✅ CMI complementarity filtering: {original_count} → {filtered_count} interactions")
                                    tprint(f"📊 Noise floor: {cmi_result.noise_floor:.6f}")
                                    tprint(f"📊 ΔPerf threshold: {cmi_result.delta_perf_threshold:.6f}")
                                    self.logger.info(f"✅ CMI complementarity filtering: {original_count} → {filtered_count} interactions")
                                    self.logger.info(f"📊 Noise floor: {cmi_result.noise_floor:.6f}")
                                    self.logger.info(f"📊 ΔPerf threshold: {cmi_result.delta_perf_threshold:.6f}")
                                    
                                    # Store CMI diagnostics in metadata
                                    interaction_metadata['cmi_diagnostics'] = {
                                        'cmi_enabled': True,
                                        'original_interactions': original_count,
                                        'filtered_interactions': filtered_count,
                                        'noise_floor': cmi_result.noise_floor,
                                        'delta_perf_threshold': cmi_result.delta_perf_threshold,
                                        'analyst_source': analyst_result.source,
                                        'analyst_dims': analyst_result.n_dims,
                                        'I_Y_A': analyst_result.I_Y_A,
                                        'degraded_to_unconditional': analyst_result.degraded_to_unconditional
                                    }
                                else:
                                    tprint("⚠️ CMI complementarity scoring failed for interactions, using all interactions")
                                    self.logger.warning("⚠️ CMI complementarity scoring failed for interactions, using all interactions")
                                    interaction_metadata['cmi_diagnostics'] = {'cmi_enabled': False, 'error': 'CMI scoring failed'}
                            else:
                                tprint("⚠️ Analyst side information extraction failed for interactions, using all interactions")
                                self.logger.warning("⚠️ Analyst side information extraction failed for interactions, using all interactions")
                                interaction_metadata['cmi_diagnostics'] = {'cmi_enabled': False, 'error': 'Analyst side info failed'}
                        else:
                            tprint("⚠️ No targets available for CMI complementarity filtering")
                            self.logger.warning("⚠️ No targets available for CMI complementarity filtering")
                            interaction_metadata['cmi_diagnostics'] = {'cmi_enabled': False, 'error': 'No targets available'}
                            
                    except Exception as e:
                        tprint(f"⚠️ CMI complementarity filtering failed for interactions: {e}, using all interactions")
                        self.logger.warning(f"⚠️ CMI complementarity filtering failed for interactions: {e}, using all interactions")
                        interaction_metadata['cmi_diagnostics'] = {'cmi_enabled': False, 'error': str(e)}
                else:
                    tprint("📊 CMI complementarity not enabled or no interactions available")
                    interaction_metadata['cmi_diagnostics'] = {'cmi_enabled': False, 'reason': 'Not in Tactician mode or no interactions'}
                
                tprint("💾 Storing interaction features in artifact manager")
                artifact_manager.store_enhanced(ArtifactKeys.INTERACTION_FEATURES, interaction_features, {
                    'step': 'interaction_generation',
                    'shape': interaction_features.shape if hasattr(interaction_features, 'shape') else None,
                    'created_at': datetime.now().isoformat()
                })
                
                tprint("💾 Storing interaction metadata in artifact manager")
                artifact_manager.store_enhanced(ArtifactKeys.INTERACTION_METADATA, interaction_metadata, {
                    'step': 'interaction_generation',
                    'created_at': datetime.now().isoformat()
                })
                
                tprint("💾 Storing generation metrics in artifact manager")
                artifact_manager.store_enhanced(ArtifactKeys.INTERACTION_GENERATION_METRICS, generation_metrics, {
                    'step': 'interaction_generation',
                    'created_at': datetime.now().isoformat()
                })

            tprint("📊 Creating InteractionGenerationResult object")
            result_obj = InteractionGenerationResult(
                success=bool(result.get('success', False)),
                interaction_features=result.get('interaction_features', pd.DataFrame()),
                interaction_metadata=result.get('interaction_metadata', {}),
                generation_metrics=result.get('generation_metrics', {}),
                artifacts=result.get('artifacts', {}),
                error_message=result.get('error_message')
            )
            tprint(f"📊 Result object created: success={result_obj.success}, features_shape={result_obj.interaction_features.shape if hasattr(result_obj.interaction_features, 'shape') else 'None'}")
            
            # Build human-readable report
            tprint("📊 Generating interaction report")
            try:
                report = self._generate_interaction_report(
                    result_obj.interaction_features,
                    result_obj.interaction_metadata,
                    symbol,
                    timeframe,
                    data
                )
                md = self._format_interaction_markdown(report)
                self._store_interaction_report(report, md, symbol, timeframe)
                tprint("📊 Report generated and stored successfully")
            except Exception as e:
                tprint(f"⚠️ Report generation failed: {e}")
                pass
            tprint("✅ Returning result object")
            return result_obj
        except Exception as e:
            tprint(f"❌ Interaction generation failed: {e}")
            self.logger.error(f"Interaction generation failed: {e}")
            return InteractionGenerationResult(
                success=False,
                interaction_features=pd.DataFrame(),
                interaction_metadata={},
                generation_metrics={},
                artifacts={},
                error_message=str(e)
            )

    # --- Reporting helpers ---
    def _generate_interaction_report(self, interactions: pd.DataFrame, metadata: Dict[str, Any], symbol: str, timeframe: str, raw_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        from datetime import datetime as _dt
        import numpy as _np
        import pandas as _pd

        n_rows = int(len(interactions)) if isinstance(interactions, _pd.DataFrame) else 0
        n_cols = int(len(interactions.columns)) if isinstance(interactions, _pd.DataFrame) else 0

        # Proxy target
        corr_rows = []
        if isinstance(raw_data, _pd.DataFrame) and 'close' in raw_data.columns and isinstance(interactions, _pd.DataFrame) and not interactions.empty:
            returns = raw_data['close'].pct_change().fillna(0.0)
            # Align and sample
            df = _pd.concat([interactions, returns.rename('ret')], axis=1).dropna()
            if not df.empty:
                if len(df) > 200_000:
                    df = df.iloc[-200_000:]
                y = df['ret'].values
                def safe_corr(xv, yv):
                    try:
                        xv = _np.asarray(xv)
                        yv = _np.asarray(yv)
                        xv = xv - xv.mean()
                        yv = yv - yv.mean()
                        denom = (_np.sqrt((xv*xv).sum()) * _np.sqrt((yv*yv).sum()))
                        return float((xv*yv).sum() / denom) if denom != 0 else 0.0
                    except Exception:
                        return 0.0
                cols = interactions.columns[:200]
                for c in cols:
                    try:
                        x = df[c].values
                        corr = abs(safe_corr(x, y))
                        nn = (df[c].notna().sum() / len(df)) * 100.0
                        var = float(_np.nanvar(df[c].values))
                        corr_rows.append({'feature': c, 'abs_corr_ret': round(corr, 6), 'non_null_pct': round(nn,2), 'variance': round(var, 6)})
                    except Exception:
                        continue
        # Sort by |corr|
        corr_rows = sorted(corr_rows, key=lambda d: d['abs_corr_ret'], reverse=True)[:40]

        return {
            'title': 'Interaction Generation Report',
            'timestamp': _dt.now().isoformat(),
            'configuration': {'symbol': symbol, 'timeframe': timeframe},
            'summary': {
                'rows': n_rows,
                'columns': n_cols,
                'memory_mb': float(interactions.memory_usage(deep=True).sum() / (1024**2)) if isinstance(interactions, _pd.DataFrame) else 0.0
            },
            'cmi_diagnostics': (metadata or {}).get('cmi_diagnostics', {}),
            'top_interactions': corr_rows
        }

    def _format_interaction_markdown(self, report: Dict[str, Any]) -> str:
        md = f"# {report['title']}\n\n"
        md += f"**Generated:** {report['timestamp']}\n\n"
        cfg = report.get('configuration', {})
        md += "## 📌 Configuration\n\n"
        md += f"- Symbol: {cfg.get('symbol','?')}\n"
        md += f"- Timeframe: {cfg.get('timeframe','?')}\n"

        summ = report.get('summary', {})
        md += "\n## 📊 Summary\n\n"
        md += f"- Rows: {summ.get('rows',0):,}\n"
        md += f"- Interactions: {summ.get('columns',0)}\n"
        md += f"- Memory: {summ.get('memory_mb',0.0):.2f} MB\n"

        md += "\n## 🔝 Top Interactions by |Corr| vs returns\n\n"
        if report.get('top_interactions'):
            md += "| Feature | |Corr| | Non-Null % | Variance |\n|---|---:|---:|---:|\n"
            for r in report['top_interactions']:
                md += f"| {r['feature']} | {r['abs_corr_ret']:.4f} | {r['non_null_pct']:.2f} | {r['variance']:.6f} |\n"
        else:
            md += "_Correlation not computed (missing close data).\n_"

        # CMI section
        md += "\n## 🧠 CMI Diagnostics\n\n"
        cmi = report.get('cmi_diagnostics', {})
        if cmi:
            for k, v in cmi.items():
                md += f"- {k}: {v}\n"
        else:
            md += "- Not available\n"
        return md

    def _store_interaction_report(self, report: Dict[str, Any], markdown: str, symbol: str, timeframe: str) -> None:
        from datetime import datetime as _dt
        from pathlib import Path as _Path
        import json as _json
        out_dir = _Path('outcomes')
        out_dir.mkdir(exist_ok=True)
        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
        md_path = out_dir / f"interaction_generation_report_{symbol}_{timeframe}_{ts}.md"
        json_path = out_dir / f"interaction_generation_report_{symbol}_{timeframe}_{ts}.json"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        with open(json_path, 'w', encoding='utf-8') as f:
            _json.dump(report, f, indent=2, ensure_ascii=False)

    # Minimal hooks for ModularComponent
    def _initialize_resources(self) -> bool:
        try:
            self.set_state('initialized', True)
            return True
        except Exception:
            return False

    def _cleanup_resources(self) -> None:
        self.set_state('initialized', False)

    def _process_data(self, data: Any, **kwargs) -> Any:
        return data

    def _get_validation_rules(self) -> Dict[str, Any]:
        return {
            'data_types': ['pandas.DataFrame'],
            'required_attributes': ['open', 'high', 'low', 'close', 'volume'],
            'min_size': 100
        }

    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        errors, warnings, metadata = [], [], {}
        if isinstance(data, pd.DataFrame):
            missing = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c not in data.columns]
            if missing:
                errors.append(f"Missing required columns: {missing}")
            metadata['shape'] = data.shape
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}


# Handler for ares_launcher/sub_pipeline integration
async def handle_feature_generation_interaction_generation_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    data: Optional[pd.DataFrame] = None,
    **kwargs
) -> InteractionGenerationResult:
    """Execute interaction generation via consolidated pipeline runner."""
    tprint("🔧 Starting handle_feature_generation_interaction_generation_step")
    tprint(f"📊 Handler params: symbol={symbol}, timeframe={timeframe}, direction={direction}, intensity={intensity}")
    tprint(f"📊 Data params: data_shape={data.shape if hasattr(data, 'shape') else 'None'}, lookback_days={lookback_days}, start_date={start_date}, end_date={end_date}")
    
    manager = get_pretraining_artifact_manager()
    tprint("📦 Got pretraining artifact manager")

    # Attempt to lazily load data if not provided
    tprint("🔍 Attempting to load selected features from artifact manager")
    # Enforce using only selected features
    data = manager.get_dataframe('feature_selection', ArtifactKeys.SELECTED_FEATURES)
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        tprint("🔍 Trying alternative selection step key")
        data = manager.get_dataframe('feature_generation_feature_selection_step', ArtifactKeys.SELECTED_FEATURES)
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return InteractionGenerationResult(
            success=False,
            interaction_features=pd.DataFrame(),
            interaction_metadata={},
            generation_metrics={},
            artifacts={},
            error_message="Selected features not found. Run feature_selection before interaction_generation."
        )

    tprint("🚀 Calling run_interaction_generation_step from handler")
    # Load targets from artifact manager for runner
    try:
        precomp_targets = None
        for step_name in ("labeling_integration", "feature_generation_labeling_integration_step"):
            for key in ("targets", ArtifactKeys.TARGETS):
                tmp = manager.get_artifact(step_name, key)
                if isinstance(tmp, pd.Series) and not tmp.empty:
                    precomp_targets = tmp
                    break
                if isinstance(tmp, pd.DataFrame) and not tmp.empty:
                    precomp_targets = tmp.iloc[:, 0]
                    break
            if isinstance(precomp_targets, pd.Series) and not precomp_targets.empty:
                break
    except Exception:
        precomp_targets = None

    result_dict = await run_interaction_generation_step(
        data=data,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        intensity=intensity,
        lookback_days=lookback_days,
        start_date=start_date,
        end_date=end_date,
        exchange=exchange,
        custom_overrides={'targets': precomp_targets} if isinstance(precomp_targets, pd.Series) and not precomp_targets.empty else custom_overrides
    )
    tprint(f"✅ run_interaction_generation_step completed: success={result_dict.get('success', False)}")

    tprint("📊 Creating InteractionGenerationResult from handler")
    result = InteractionGenerationResult(
        success=bool(result_dict.get('success', False)),
        interaction_features=result_dict.get('interaction_features', pd.DataFrame()),
        interaction_metadata=result_dict.get('interaction_metadata', {}),
        generation_metrics=result_dict.get('generation_metrics', {}),
        artifacts=result_dict.get('artifacts', {}),
        error_message=result_dict.get('error_message')
    )
    tprint(f"📊 Handler result: success={result.success}, features_shape={result.interaction_features.shape if hasattr(result.interaction_features, 'shape') else 'None'}")

    if result.success:
        tprint("💾 Saving artifacts to manager")
        result.artifacts.setdefault(ArtifactKeys.INTERACTION_FEATURES, result.interaction_features)
        manager.save('feature_generation_interaction_generation_step', result.artifacts, metadata=result.interaction_metadata)
        tprint("✅ Artifacts saved successfully")

    tprint("✅ Handler completed, returning result")
    return result
