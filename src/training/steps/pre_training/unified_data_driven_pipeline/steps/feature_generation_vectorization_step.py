"""
Enhanced Feature Vectorization Step

This step exposes feature vectorization via the unified consolidated pipeline,
returning a lightweight result with vectorized features and performance metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from tprint import tprint

from src.training.steps.pre_training.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent
)
from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import (
    run_vectorization_step
)
from src.training.steps.pre_training.utils.artifact_manager import (
    get_pretraining_artifact_manager,
    ArtifactKeys,
)


@dataclass
class VectorizationResult:
    success: bool
    vectorized_features: pd.DataFrame
    vectorization_metadata: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class FeatureGenerationVectorizationStep(ModularComponent):
    """Vectorization step backed by the consolidated pipeline runner."""

    def __init__(self, name: str = "vectorization_step",
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(name, config or {}, logger)

    async def execute(self,
                      training_input: Dict[str, Any],
                      pipeline_state: Dict[str, Any]) -> VectorizationResult:
        """Execute vectorization using the consolidated pipeline runner with artifact manager integration."""
        tprint("🚀 Starting vectorization step execution")
        tprint(f"📥 Training input keys: {list(training_input.keys())}")
        tprint(f"📊 Pipeline state keys: {list(pipeline_state.keys())}")
        
        # Get artifact manager
        artifact_manager = get_pretraining_artifact_manager()
        tprint("📦 Retrieved artifact manager")
        
        # Try to load from artifact manager first (pretraining store)
        tprint("🔍 Checking for cached vectorized features...")
        cached_vectorized = artifact_manager.get_artifact('vectorization', 'vectorized_features')
        cached_metadata = artifact_manager.get_artifact('vectorization', 'vectorization_metadata')
        cached_metrics = artifact_manager.get_artifact('vectorization', 'vectorization_metrics')
        # Backwards-compatible step name
        if cached_vectorized is None:
            cached_vectorized = artifact_manager.get_artifact('feature_generation_vectorization_step', 'vectorized_features')
            cached_metadata = cached_metadata or artifact_manager.get_artifact('feature_generation_vectorization_step', 'vectorization_metadata')
            cached_metrics = cached_metrics or artifact_manager.get_artifact('feature_generation_vectorization_step', 'vectorization_metrics')
        # Final fallback: enhanced artifact manager cache
        if cached_vectorized is None:
            try:
                from src.utils.artifact_manager import ArtifactManager as _EnhancedAM
                _enh = _EnhancedAM(config={})
                cached_vectorized = _enh.retrieve_enhanced(ArtifactKeys.VECTORIZED_FEATURES)
                cached_metadata = cached_metadata or _enh.retrieve_enhanced(ArtifactKeys.VECTORIZATION_METADATA)
                cached_metrics = cached_metrics or _enh.retrieve_enhanced(ArtifactKeys.VECTORIZATION_METRICS)
            except Exception:
                pass
        
        if cached_vectorized is not None:
            tprint("✅ Found cached vectorized features - returning cached result")
            tprint(f"📊 Cached features shape: {cached_vectorized.shape if hasattr(cached_vectorized, 'shape') else 'Unknown'}")
            self.logger.info("📦 Retrieved vectorized features from artifact manager")
            return VectorizationResult(
                success=True,
                vectorized_features=cached_vectorized,
                vectorization_metadata=cached_metadata or {},
                performance_metrics=cached_metrics or {},
                artifacts={'cache_hit': True},
                error_message=None
            )
        else:
            tprint("❌ No cached vectorized features found - proceeding with computation")

        data = training_input.get('data')
        tprint(f"📊 Input data type: {type(data)}")
        tprint(f"📊 Input data shape: {data.shape if hasattr(data, 'shape') else 'No shape attribute'}")
        tprint(f"📊 Input data empty: {data.empty if hasattr(data, 'empty') else 'No empty attribute'}")
        
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            tprint("🔍 Auto-loading features for vectorization")
            period_features = artifact_manager.get_dataframe(
                'feature_generation_period_lookback_optimization_step',
                ArtifactKeys.OPTIMIZED_FEATURE_DATAFRAME
            )
            interaction_features = artifact_manager.get_dataframe(
                'feature_generation_interaction_generation_step',
                ArtifactKeys.INTERACTION_FEATURES
            )

            frames = []
            if isinstance(period_features, pd.DataFrame) and not period_features.empty:
                tprint(f"📊 Loaded period/lookback features: shape={period_features.shape}")
                frames.append(period_features)
            if isinstance(interaction_features, pd.DataFrame) and not interaction_features.empty:
                tprint(f"📊 Loaded interaction features: shape={interaction_features.shape}")
                frames.append(interaction_features)

            if frames:
                data = pd.concat(frames, axis=1, join='outer')
                data = data.loc[:, ~data.columns.duplicated()]
                data = data.dropna(axis=0, how='all')
                tprint(f"✅ Combined period + interaction features: shape={data.shape}")

        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            tprint("⚠️ Period + interaction features unavailable; falling back to selected/generation artifacts")
            data = artifact_manager.get_dataframe('feature_generation_feature_selection_step', ArtifactKeys.SELECTED_FEATURES)
            if data is None or data.empty:
                data = artifact_manager.get_dataframe('feature_generation_feature_generation_step', ArtifactKeys.FEATURE_DATAFRAME)
            if data is None or not isinstance(data, pd.DataFrame) or data.empty:
                tprint("❌ Invalid input data - returning error result")
                return VectorizationResult(
                    success=False,
                    vectorized_features=pd.DataFrame(),
                    vectorization_metadata={},
                    performance_metrics={},
                    artifacts={},
                    error_message="Input 'data' must be a non-empty pandas DataFrame."
                )

        symbol = training_input.get('symbol', 'ETHUSDT')
        timeframe = training_input.get('timeframe', '15m')
        direction = training_input.get('direction', 'longs')
        intensity = training_input.get('intensity', 'blank')
        lookback_days = training_input.get('lookback_days')
        start_date = training_input.get('start_date')
        end_date = training_input.get('end_date')
        exchange = training_input.get('exchange', 'binance')
        custom_overrides = training_input.get('custom_overrides')
        
        tprint(f"⚙️ Configuration: symbol={symbol}, timeframe={timeframe}, direction={direction}")
        tprint(f"⚙️ Additional params: intensity={intensity}, lookback_days={lookback_days}")
        tprint(f"⚙️ Date range: {start_date} to {end_date}, exchange={exchange}")
        tprint(f"⚙️ Custom overrides: {custom_overrides is not None}")

        try:
            tprint("🔄 Calling run_vectorization_step...")
            result_dict = await run_vectorization_step(
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
            tprint(f"✅ run_vectorization_step completed successfully")
            tprint(f"📊 Result keys: {list(result_dict.keys()) if result_dict else 'None'}")

            # Store artifacts in artifact manager
            if result_dict.get('success', False):
                tprint("💾 Storing successful vectorization results in artifact manager...")
                vectorized_features = result_dict.get('vectorized_features', pd.DataFrame())
                vectorization_metadata = result_dict.get('vectorization_metadata', {})
                performance_metrics = result_dict.get('performance_metrics', {})
                
                tprint(f"📊 Vectorized features shape: {vectorized_features.shape if hasattr(vectorized_features, 'shape') else 'Unknown'}")
                tprint(f"📊 Metadata keys: {list(vectorization_metadata.keys()) if vectorization_metadata else 'None'}")
                tprint(f"📊 Performance metrics keys: {list(performance_metrics.keys()) if performance_metrics else 'None'}")
                
                artifact_manager.save(
                    step_name='vectorization',
                    artifacts={
                        'vectorized_features': vectorized_features,
                        'vectorization_metadata': vectorization_metadata,
                        'vectorization_metrics': performance_metrics
                    },
                    metadata={
                        'step': 'vectorization',
                        'shape': vectorized_features.shape if hasattr(vectorized_features, 'shape') else None,
                        'created_at': datetime.now().isoformat()
                    }
                )
                artifact_manager.save(
                    step_name='feature_generation_vectorization_step',
                    artifacts={
                        'vectorized_features': vectorized_features,
                        'vectorization_metadata': vectorization_metadata,
                        'vectorization_metrics': performance_metrics
                    },
                    metadata={
                        'step': 'feature_generation_vectorization_step',
                        'shape': vectorized_features.shape if hasattr(vectorized_features, 'shape') else None,
                        'created_at': datetime.now().isoformat()
                    }
                )
                tprint("✅ Stored vectorized features and metadata in artifact manager")
            else:
                tprint("❌ Vectorization step failed - not storing artifacts")

            tprint("🎯 Creating VectorizationResult...")
            result = VectorizationResult(
                success=bool(result_dict.get('success', False)),
                vectorized_features=result_dict.get('vectorized_features', pd.DataFrame()),
                vectorization_metadata=result_dict.get('vectorization_metadata', {}),
                performance_metrics=result_dict.get('performance_metrics', {}),
                artifacts=result_dict.get('artifacts', {}),
                error_message=result_dict.get('error_message')
            )
            tprint(f"✅ VectorizationResult created - success: {result.success}")
            return result
        except Exception as e:
            tprint(f"💥 Exception in vectorization execution: {e}")
            tprint(f"💥 Exception type: {type(e).__name__}")
            self.logger.error(f"Vectorization execution failed: {e}")
            return VectorizationResult(
                success=False,
                vectorized_features=pd.DataFrame(),
                vectorization_metadata={},
                performance_metrics={},
                artifacts={},
                error_message=str(e)
            )

    # Minimal hooks for ModularComponent
    def _initialize_resources(self) -> bool:
        tprint("🔧 Initializing vectorization step resources...")
        try:
            self.set_state('initialized', True)
            tprint("✅ Vectorization step resources initialized successfully")
            return True
        except Exception as e:
            tprint(f"❌ Failed to initialize vectorization step resources: {e}")
            return False

    def _cleanup_resources(self) -> None:
        tprint("🧹 Cleaning up vectorization step resources...")
        self.set_state('initialized', False)
        tprint("✅ Vectorization step resources cleaned up")

    def _process_data(self, data: Any, **kwargs) -> Any:
        tprint(f"🔄 Processing data in vectorization step: {type(data)}")
        tprint(f"📊 Data shape: {data.shape if hasattr(data, 'shape') else 'No shape'}")
        return data

    def _get_validation_rules(self) -> Dict[str, Any]:
        return {
            'data_types': ['pandas.DataFrame'],
            'required_attributes': ['open', 'high', 'low', 'close', 'volume'],
            'min_size': 100
        }

    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        tprint("🔍 Validating vectorization step component-specific requirements...")
        errors, warnings, metadata = [], [], {}
        if isinstance(data, pd.DataFrame):
            tprint(f"📊 Validating DataFrame with shape: {data.shape}")
            missing = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c not in data.columns]
            if missing:
                tprint(f"❌ Missing required columns: {missing}")
                errors.append(f"Missing required columns: {missing}")
            else:
                tprint("✅ All required columns present")
            metadata['shape'] = data.shape
            tprint(f"📊 Validation metadata: {metadata}")
        else:
            tprint(f"❌ Data is not a DataFrame: {type(data)}")
            errors.append(f"Data must be a pandas DataFrame, got {type(data)}")
        
        result = {'errors': errors, 'warnings': warnings, 'metadata': metadata}
        tprint(f"🔍 Validation result: {len(errors)} errors, {len(warnings)} warnings")
        return result

    # --- Reporting helpers ---
    def _generate_vectorization_report(self, X: pd.DataFrame, meta: Dict[str, Any], perf: Dict[str, Any], symbol: str, timeframe: str) -> Dict[str, Any]:
        tprint("📝 Generating vectorization report...")
        from datetime import datetime as _dt
        import pandas as _pd
        rows = int(len(X)) if isinstance(X, _pd.DataFrame) else 0
        cols = int(len(X.columns)) if isinstance(X, _pd.DataFrame) else 0
        mem = float(X.memory_usage(deep=True).sum() / (1024**2)) if isinstance(X, _pd.DataFrame) else 0.0
        tprint(f"📊 Report data: {rows} rows, {cols} columns, {mem:.2f} MB")
        
        sparsity = 0.0
        if isinstance(X, _pd.DataFrame) and rows and cols:
            sparsity = float((X.isna().sum().sum()) / (rows * cols)) * 100.0
            tprint(f"📊 Sparsity: {sparsity:.2f}%")
        
        # Top variance
        top_vars = []
        if isinstance(X, _pd.DataFrame) and not X.empty:
            try:
                tprint("🔍 Calculating feature variances...")
                v = X.var(numeric_only=True).sort_values(ascending=False)
                for name, val in v.head(40).items():
                    top_vars.append({'feature': name, 'variance': float(val) if val == val else 0.0})
                tprint(f"📊 Calculated variances for {len(top_vars)} features")
            except Exception as e:
                tprint(f"❌ Failed to calculate variances: {e}")
        
        report = {
            'title': 'Vectorization Report',
            'timestamp': _dt.now().isoformat(),
            'configuration': {'symbol': symbol, 'timeframe': timeframe},
            'summary': {'rows': rows, 'columns': cols, 'memory_mb': mem, 'sparsity_pct': sparsity},
            'metadata': meta or {},
            'performance_metrics': perf or {},
            'top_variance_features': top_vars
        }
        tprint("✅ Vectorization report generated successfully")
        return report

    def _format_vectorization_markdown(self, report: Dict[str, Any]) -> str:
        md = f"# {report['title']}\n\n"
        md += f"**Generated:** {report['timestamp']}\n\n"
        cfg = report.get('configuration', {})
        md += "## 📌 Configuration\n\n"
        md += f"- Symbol: {cfg.get('symbol','?')}\n"
        md += f"- Timeframe: {cfg.get('timeframe','?')}\n"
        summ = report.get('summary', {})
        md += "\n## 📊 Summary\n\n"
        md += f"- Rows: {summ.get('rows',0):,}\n"
        md += f"- Columns: {summ.get('columns',0)}\n"
        md += f"- Memory: {summ.get('memory_mb',0.0):.2f} MB\n"
        md += f"- Sparsity: {summ.get('sparsity_pct',0.0):.2f}%\n"
        # Performance/metadata sections
        def _dump(title, obj):
            nonlocal md
            if obj:
                md += f"\n## {title}\n\n"
                for k, v in obj.items():
                    md += f"- {k}: {v}\n"
        _dump('⚙️ Metadata', report.get('metadata', {}))
        _dump('⏱️ Performance', report.get('performance_metrics', {}))

        # Top variance
        md += "\n## 🔝 Top Features by Variance\n\n"
        if report.get('top_variance_features'):
            md += "| Feature | Variance |\n|---|---:|\n"
            for r in report['top_variance_features']:
                md += f"| {r['feature']} | {r['variance']:.6f} |\n"
        else:
            md += "_No variance information available._\n"
        return md

    def _store_vectorization_report(self, report: Dict[str, Any], markdown: str, symbol: str, timeframe: str) -> None:
        tprint("💾 Storing vectorization report...")
        from datetime import datetime as _dt
        from pathlib import Path as _Path
        import json as _json
        out_dir = _Path('outcomes')
        out_dir.mkdir(exist_ok=True)
        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
        md_path = out_dir / f"vectorization_report_{symbol}_{timeframe}_{ts}.md"
        json_path = out_dir / f"vectorization_report_{symbol}_{timeframe}_{ts}.json"
        
        tprint(f"📁 Report paths: MD={md_path}, JSON={json_path}")
        
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
            tprint("✅ Markdown report saved")
            
            with open(json_path, 'w', encoding='utf-8') as f:
                _json.dump(report, f, indent=2, ensure_ascii=False)
            tprint("✅ JSON report saved")
        except Exception as e:
            tprint(f"❌ Failed to save vectorization report: {e}")


# Handler for ares_launcher/sub_pipeline integration
async def handle_feature_generation_vectorization_step(
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
) -> VectorizationResult:
    """Execute the vectorization step via consolidated pipeline runner."""
    tprint("🚀 Starting handle_feature_generation_vectorization_step")
    tprint(f"📥 Handler params: symbol={symbol}, timeframe={timeframe}, direction={direction}")
    tprint(f"📥 Additional params: intensity={intensity}, lookback_days={lookback_days}")
    tprint(f"📥 Data provided: {data is not None and not data.empty if data is not None else 'None'}")
    
    manager = get_pretraining_artifact_manager()
    tprint("📦 Retrieved artifact manager in handler")

    # Attempt to lazily load data if not provided
    tprint("🔍 Attempting to load data from artifact manager...")
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        tprint("🔍 Trying to load optimized period/lookback features...")
        period_features = manager.get_dataframe('feature_generation_period_lookback_optimization_step', ArtifactKeys.OPTIMIZED_FEATURE_DATAFRAME)
        tprint(f"📊 Period features result: {period_features is not None and not period_features.empty if period_features is not None else 'None'}")

        tprint("🔍 Trying to load interaction features...")
        interaction_features = manager.get_dataframe('feature_generation_interaction_generation_step', ArtifactKeys.INTERACTION_FEATURES)
        tprint(f"📊 Interaction features result: {interaction_features is not None and not interaction_features.empty if interaction_features is not None else 'None'}")

        frames = [df for df in (period_features, interaction_features) if isinstance(df, pd.DataFrame) and not df.empty]
        if frames:
            combined = pd.concat(frames, axis=1, join='outer')
            combined = combined.loc[:, ~combined.columns.duplicated()]
            combined = combined.dropna(axis=0, how='all')
            data = combined
            tprint(f"✅ Combined period + interaction features: shape={data.shape}")

    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        tprint("⚠️ Period + interaction features unavailable; trying selected features...")
        data = manager.get_dataframe('feature_generation_feature_selection_step', ArtifactKeys.SELECTED_FEATURES)
        tprint(f"📊 Selected features result: {data is not None and not data.empty if data is not None else 'None'}")
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        tprint("⚠️ Selected features unavailable; trying generated feature set...")
        data = manager.get_dataframe('feature_generation_feature_generation_step', ArtifactKeys.FEATURE_DATAFRAME)
        tprint(f"📊 Feature generation result: {data is not None and not data.empty if data is not None else 'None'}")
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        tprint("🔍 Attempting to load data via data validation step...")
        try:
            from .feature_generation_data_validation_step import FeatureGenerationDataValidationStep  # type: ignore
            loader = FeatureGenerationDataValidationStep()
            loaded = await loader._load_data_for_validation(  # noqa: SLF001 (intentional internal use)
                symbol, timeframe, exchange, start_date, end_date, lookback_days
            )
            data = loaded
            tprint(f"📊 Data validation loader result: {data is not None and not data.empty if data is not None else 'None'}")
        except Exception as e:
            tprint(f"❌ Data validation loader failed: {e}")
            return VectorizationResult(
                success=False,
                vectorized_features=pd.DataFrame(),
                vectorization_metadata={},
                performance_metrics={},
                artifacts={},
                error_message="Input 'data' must be a non-empty pandas DataFrame (auto-load failed)."
            )

    tprint(f"📊 Final data shape before vectorization: {data.shape if hasattr(data, 'shape') else 'Unknown'}")
    tprint("🔄 Calling run_vectorization_step from handler...")
    result_dict = await run_vectorization_step(
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
    tprint(f"✅ Handler run_vectorization_step completed")
    tprint(f"📊 Handler result keys: {list(result_dict.keys()) if result_dict else 'None'}")

    tprint("🎯 Creating VectorizationResult in handler...")
    vectorization_result = VectorizationResult(
        success=bool(result_dict.get('success', False)),
        vectorized_features=result_dict.get('vectorized_features', pd.DataFrame()),
        vectorization_metadata=result_dict.get('vectorization_metadata', {}),
        performance_metrics=result_dict.get('performance_metrics', {}),
        artifacts=result_dict.get('artifacts', {}),
        error_message=result_dict.get('error_message')
    )
    tprint(f"✅ Handler VectorizationResult created - success: {vectorization_result.success}")

    if vectorization_result.success:
        tprint("💾 Saving successful vectorization results in handler...")
        vectorization_result.artifacts.setdefault(ArtifactKeys.VECTORIZED_FEATURES, vectorization_result.vectorized_features)
        manager.save('feature_generation_vectorization_step', vectorization_result.artifacts, metadata=vectorization_result.vectorization_metadata)
        tprint("✅ Handler artifacts saved successfully")
    else:
        tprint("❌ Handler vectorization failed - not saving artifacts")

    # Best-effort outcomes report
    tprint("📝 Generating vectorization report...")
    try:
        step = FeatureGenerationVectorizationStep()
        report = step._generate_vectorization_report(
            vectorization_result.vectorized_features,
            vectorization_result.vectorization_metadata,
            vectorization_result.performance_metrics,
            symbol,
            timeframe
        )
        md = step._format_vectorization_markdown(report)
        step._store_vectorization_report(report, md, symbol, timeframe)
        tprint("✅ Vectorization report generated and stored")
    except Exception as e:
        tprint(f"❌ Failed to generate vectorization report: {e}")

    tprint("🏁 Handler vectorization step completed")
    return vectorization_result

    # --- Reporting helpers (shared with execute via handler instance) ---
    
