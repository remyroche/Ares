import sys
from pathlib import Path

from src.training.steps.market_analysis import regime_analysis_script


def test_cli_main_smoke(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    data_cache = tmp_path / "data_cache"
    data_cache.mkdir()

    captured = {}

    class DummyService:
        def __init__(self, data_cache_path):
            captured["data_cache_path"] = Path(data_cache_path)

        def analyze(self, symbol):
            captured["symbol"] = symbol
            return {}

    monkeypatch.setattr(regime_analysis_script, "RegimeAnalysisService", DummyService)
    monkeypatch.setattr(regime_analysis_script, "tprint", lambda *args, **kwargs: None)

    monkeypatch.setattr(sys, "argv", [
        "regime_analysis_script",
        "--symbol",
        "BTCUSDT",
        "--data-cache",
        str(data_cache),
    ])

    result = regime_analysis_script.main()

    assert result == 0
    assert captured["symbol"] == "BTCUSDT"
    assert captured["data_cache_path"] == data_cache
