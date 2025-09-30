import numpy as np

from src.training.steps.market_analysis.regime_analysis.service import RegimeAnalysisService
from src.training.steps.market_analysis.regime_analysis import service as service_module


def test_service_analyze_creates_result_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    data_cache = tmp_path / "data_cache"
    data_cache.mkdir()

    nas_features = np.array([[0.0, 0.1], [0.1, 0.2], [1.0, 1.1], [1.1, 1.2]])
    nas_labels = np.array([0, 0, 1, 1])
    tas_features = np.array([[0.2, 0.3], [0.3, 0.4], [1.2, 1.3], [1.3, 1.4]])
    tas_labels = np.array([0, 0, 1, 1])

    monkeypatch.setattr(
        service_module,
        "load_regime_datasets",
        lambda *_: (nas_features, nas_labels, tas_features, tas_labels),
    )

    analysis_service = RegimeAnalysisService(data_cache_path=data_cache)
    analysis = analysis_service.analyze(symbol="ETHUSDT")

    assert analysis["summary"]["nas_samples"] == 4
    assert any(path.name.startswith("ETHUSDT_regime_analysis_") for path in (tmp_path / "regime_analysis_results").iterdir())
