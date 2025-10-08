import asyncio
import sys
import types
from typing import Any, Dict, List, Optional

import pandas as pd

from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent,
    ComponentConfig,
    ComponentResult,
)


class _RecordingArtifactManager:
    def __init__(self) -> None:
        self.saved_payloads: List[Dict[str, Any]] = []

    def save_artifact(self, data: Any, base_name: str, extension: str = ".json", **_: Any) -> str:
        self.saved_payloads.append({
            'base_name': base_name,
            'data': data,
        })
        return f"/tmp/{base_name}{extension}"


class _StubComponent(BasePreTrainingComponent):
    def __init__(self, step_name: str, config: Optional[ComponentConfig] = None):
        super().__init__(config)
        self.step_name = step_name

    def get_required_artifacts(self) -> List[str]:
        return [f"{self.step_name}_artifact"]

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        artifacts = self._build_artifact_payload()
        await self.save_artifacts(artifacts, {'component_type': self.step_name})

        result = ComponentResult(
            success=True,
            artifacts=artifacts,
            metadata={'component_type': self.step_name}
        )
        # Ensure interactive step compatibility
        setattr(result, 'output_files', [f"{self.step_name}.json"])
        return result

    def _build_artifact_payload(self) -> Dict[str, Any]:
        if self.step_name == 'multi_horizon_profit_labeler':
            return {
                'multi_horizon_labeling_result': {
                    'labeled_data': pd.DataFrame({'label': [1]})
                }
            }
        if self.step_name == 'feature_lookback_optimization':
            return {
                'feature_lookback_optimization_result': {
                    'optimized_features': {'feature_a': 1.0}
                }
            }
        if self.step_name == 'interactive_feature_generation':
            return {
                'interactive_feature_generation_result': {
                    'features': {'f1': 1.0}
                }
            }
        if self.step_name == 'final_feature_selection':
            return {
                'final_feature_selection_result': {
                    'selected_features': ['f1', 'f2']
                }
            }
        return {f'{self.step_name}_result': {'status': 'ok'}}


def test_pre_training_pipeline_emits_run_metadata(monkeypatch):
    recorded_messages: List[str] = []

    # Inject stub component factory before importing the pipeline module
    stub_factory_module = types.ModuleType('component_factory')

    class _Factory:
        @classmethod
        def create_component(cls, name: str, config: Optional[ComponentConfig] = None) -> _StubComponent:  # type: ignore[name-defined]
            return _StubComponent(name, config)

    stub_factory_module.ComponentFactory = _Factory  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        'src.training.steps.pre_training.components.component_factory',
        stub_factory_module,
    )

    stub_interactive_module = types.ModuleType('interactive_feature_generation_component')

    class _InteractiveConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.symbol = kwargs.get('symbol')
            self.exchange = kwargs.get('exchange')
            self.timeframe = kwargs.get('timeframe')
            self.data_dir = kwargs.get('data_dir')

    def _create_interactive_component(config=None):
        return _StubComponent('interactive_feature_generation', config)

    stub_interactive_module.InteractiveFeatureGenerationConfig = _InteractiveConfig  # type: ignore[attr-defined]
    stub_interactive_module.create_interactive_feature_generation_component = _create_interactive_component  # type: ignore[attr-defined]

    monkeypatch.setitem(
        sys.modules,
        'src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.interactive_feature_generation_component',
        stub_interactive_module,
    )

    from src.training.steps.pre_training.sub_pipeline import PreTrainingSubPipeline, SubPipelineConfig

    sub_pipeline_module = sys.modules['src.training.steps.pre_training.sub_pipeline']

    def _capture_tprint(message: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        recorded_messages.append(str(message))

    # Capture tprint output for verification
    monkeypatch.setattr('src.training.steps.pre_training.sub_pipeline.tprint', _capture_tprint)

    # Deterministic run metadata sources
    monkeypatch.setattr(PreTrainingSubPipeline, '_get_git_sha', lambda self: 'test-sha')
    monkeypatch.setattr(PreTrainingSubPipeline, '_get_host_name', lambda self: 'test-host')
    monkeypatch.setattr(PreTrainingSubPipeline, '_compute_config_hash', lambda self, cfg: 'cfg-hash')

    # Stub artifact manager to capture persisted payloads
    artifact_manager = _RecordingArtifactManager()
    monkeypatch.setattr(
        'src.training.steps.pre_training.components.base_component.get_artifact_manager',
        lambda: artifact_manager,
    )

    # Replace component factory creation with our stub components
    def _create_stub_component(cls, name: str, config: Optional[ComponentConfig] = None) -> _StubComponent:  # type: ignore[override]
        return _StubComponent(name, config)

    monkeypatch.setattr(sub_pipeline_module.ComponentFactory, 'create_component', classmethod(_create_stub_component))

    pipeline = PreTrainingSubPipeline()
    config = SubPipelineConfig(
        symbol='TEST',
        exchange='stub',
        custom_params={'rng_seed': 42, 'data_snapshot_id': 'snapshot-123'},
    )

    result = asyncio.run(pipeline.execute_pipeline(config))

    assert result['success'] is True
    run_metadata = result['run_metadata']
    assert run_metadata['git_sha'] == 'test-sha'
    assert run_metadata['config_hash'] == 'cfg-hash'
    assert run_metadata['data_snapshot_id'] == 'snapshot-123'
    assert run_metadata['rng_seed'] == 42
    assert run_metadata['host_name'] == 'test-host'
    assert run_metadata['end_timestamp'] is not None
    assert run_metadata['duration_seconds'] >= 0

    # Metadata block should be printed at start and completion
    assert any('Run metadata snapshot (start)' in message for message in recorded_messages)
    assert any('duration_seconds' in message for message in recorded_messages)

    # Persisted artifacts must include run metadata payloads
    assert artifact_manager.saved_payloads, 'expected artifacts to be saved with metadata'
    first_payload = artifact_manager.saved_payloads[0]['data']
    assert 'run_metadata' in first_payload
    assert first_payload['run_metadata']['git_sha'] == 'test-sha'
