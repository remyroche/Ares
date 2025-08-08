import os
import asyncio
from datetime import datetime
import glob
import pytest
from src.reports.paper_trading_reporter import PaperTradingReporter

def test_generate_report(tmp_path):
    # Use a temporary directory for report output
    report_dir = tmp_path / "paper_trading"
    config = {"paper_trading_reporter": {"report_directory": str(report_dir), "export_formats": ["json", "csv", "html"]}}
    reporter = PaperTradingReporter(config)
    trade_data = {
        "side": "BUY",
        "leverage": 1.0,
        "duration": "scalping",
        "strategy": "test_strategy",
        "order_type": "market",
        "quantity": 1.0,
        "portfolio_percentage": 0.1,
        "risk_percentage": 0.01,
        "max_position_size": 0.1,
        "position_ranking": 1,
        "absolute_pnl": 10.0,
        "percentage_pnl": 0.01,
        "unrealized_pnl": 0.0,
        "realized_pnl": 10.0,
        "total_cost": 100.0,
        "total_proceeds": 110.0,
        "commission": 0.1,
        "slippage": 0.05,
        "net_pnl": 9.85,
        "symbol": "TEST",
        "exchange": "TESTEX",
        "timestamp": datetime.now().isoformat(),
    }
    market_indicators = {"rsi": 50, "macd": 0, "macd_signal": 0, "bollinger_upper": 0, "bollinger_lower": 0, "bollinger_middle": 0, "atr": 0, "volume_sma": 0, "price_sma_20": 0, "price_sma_50": 0, "price_sma_200": 0, "volatility": 0, "momentum": 0, "support_level": 0, "resistance_level": 0}
    market_health = {"overall_health_score": 1.0, "volatility_regime": "normal", "liquidity_score": 1.0, "stress_score": 0.0, "market_strength": 1.0, "volume_health": "good", "price_trend": "up", "market_regime": "bull"}
    ml_confidence = {"analyst_confidence": 0.9, "tactician_confidence": 0.8, "ensemble_confidence": 0.85, "meta_learner_confidence": 0.8, "individual_model_confidences": {}, "ensemble_agreement": 0.9, "model_diversity": 0.1, "prediction_consistency": 0.95}
    async def run():
        await reporter.record_trade(trade_data, market_indicators, market_health, ml_confidence)
        await reporter.generate_detailed_report("test_report", ["json", "csv", "html"])
    asyncio.run(run())
    # Assert that files are created
    files = list(report_dir.glob("paper_trading_report_*.json")) + \
            list(report_dir.glob("paper_trading_report_*.csv")) + \
            list(report_dir.glob("paper_trading_report_*.html"))
    assert len(files) == 3, f"Expected 3 report files, found {len(files)}: {files}"