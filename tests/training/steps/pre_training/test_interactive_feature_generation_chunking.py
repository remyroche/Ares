import numpy as np
import pandas as pd
import pytest
import sys
import types

from typing import Dict, Any

if 'pymc' not in sys.modules:
    pymc_stub = types.ModuleType('pymc')

    class _DummyModel:  # pragma: no cover - stub for tests
        pass

    pymc_stub.Model = _DummyModel  # type: ignore[attr-defined]
    sys.modules['pymc'] = pymc_stub

if 'aesara' not in sys.modules:
    aesara_stub = types.ModuleType('aesara')
    tensor_stub = types.ModuleType('aesara.tensor')
    sys.modules['aesara'] = aesara_stub
    sys.modules['aesara.tensor'] = tensor_stub

from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.interactive_feature_generation_component import (
    InteractiveFeatureGenerationComponent,
    OptimizedInteractionResult,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _StubOrchestrator:
    def __init__(self):
        self.calls = []
        self.config = types.SimpleNamespace(
            symbol='',
            exchange='',
            timeframe='',
            data_dir='',
        )

    async def generate_features(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> OptimizedInteractionResult:
        data = training_input['data']
        self.calls.append(data.index)
        features = pd.DataFrame({'close_feature': data['close'].values}, index=data.index)
        interaction = pd.DataFrame({'interaction': data['close'].values * 0.5}, index=data.index)
        cross = pd.DataFrame({'cross': data['close'].values * 2.0}, index=data.index)
        return OptimizedInteractionResult(
            features=features,
            feature_names=list(features.columns),
            selected_features=list(features.columns),
            interaction_features=interaction,
            cross_timeframe_features=cross,
            execution_time=0.01,
            success=True,
            artifacts={'batch_index': len(self.calls)},
        )


def _create_market_frame(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range('2024-01-01', periods=rows, freq='H')
    base = np.linspace(100, 200, rows)
    return pd.DataFrame(
        {
            'open': base,
            'high': base + 1,
            'low': base - 1,
            'close': base + 0.5,
            'volume': np.linspace(10_000, 20_000, rows),
        },
        index=index,
    )


@pytest.mark.anyio("asyncio")
async def test_interactive_component_merges_chunk_results():
    component = InteractiveFeatureGenerationComponent()
    component.orchestrator = _StubOrchestrator()

    market_data = _create_market_frame()
    targets = {'target': pd.Series(np.arange(len(market_data)), index=market_data.index)}
    pipeline_state = {'symbol': 'ETHUSDT', 'exchange': 'binance', 'timeframe': '1h'}

    full_input = {'data': market_data, 'targets': targets}
    full_result = await component.execute(full_input, pipeline_state)

    component.orchestrator = _StubOrchestrator()
    chunked_input = {
        'data': market_data,
        'data_batches': [market_data.iloc[:60], market_data.iloc[60:]],
        'targets': targets,
    }
    chunked_result = await component.execute(chunked_input, pipeline_state)

    assert chunked_result.success
    chunk_features = chunked_result.artifacts['interactive_feature_generation_result']['features']
    full_features = full_result.artifacts['interactive_feature_generation_result']['features']
    pd.testing.assert_frame_equal(chunk_features, full_features)
    assert len(component.orchestrator.calls) == 2
