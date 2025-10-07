import math
import sys
import types
from pathlib import Path


if "src.trading" not in sys.modules:
    trading_module = types.ModuleType("src.trading")
    trading_module.__path__ = [str(Path(__file__).resolve().parents[1] / "src/trading")]
    sys.modules["src.trading"] = trading_module

from src.trading.config.trading_config import TradingConfig
from src.trading.sizing.position_sizer import PositionSizer


def test_apply_position_size_modifiers_scales_with_raw_confidence_scores():
    config = TradingConfig()
    sizer = PositionSizer(config)

    base_size = 0.2
    low_confidence = sizer._apply_position_size_modifiers(base_size, 0.0, 0.0)
    mixed_confidence = sizer._apply_position_size_modifiers(base_size, 1.0, 0.0)
    high_confidence = sizer._apply_position_size_modifiers(base_size, 1.0, 1.0)

    assert math.isclose(low_confidence, base_size * 0.8, rel_tol=1e-6)
    assert math.isclose(high_confidence, base_size * 1.2, rel_tol=1e-6)
    assert low_confidence < mixed_confidence < high_confidence
