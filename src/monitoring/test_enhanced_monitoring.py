#!/usr/bin/env python3
"""
Test Script for Enhanced Monitoring System

Simple test script to verify the enhanced monitoring system functionality.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

# Import the enhanced monitoring components
from .enhanced_monitoring_orchestrator import (
    EnhancedMonitoringOrchestrator, 
    TradingMode
)
from .trade_decision_capture import TradeDecisionContextCapture
from .shap_lime_integration import ExplainabilityIntegrator
from .enhanced_ml_monitoring import (
    TradeContext, TradingIndicator, MLModelDecision, 
    EnsembleDecision, ModelType
)


class TestEnhancedMonitoring:
    """Test class for enhanced monitoring system."""
    
    def __init__(self):
        """Initialize test class."""
        self.temp_dir = None
        self.config = None
        self.orchestrator = None
        self.context_capture = None
        self.explainability_integrator = None
    
    def setup(self):
        """Setup test environment."""
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test configuration
        self.config = {
            "enhanced_monitoring": {
                "enable_monitoring": True,
                "enable_explanations": True,
                "enable_real_time_tracking": True,
                "monthly_export_enabled": True,
                "daily_export_enabled": True,
                "export_directory": str(self.temp_dir / "exports"),
                "max_decisions_in_memory": 1000,
                "data_retention_days": 30,
                "cleanup_frequency_hours": 1
            },
            "shap_analysis": {
                "enable_shap": False,  # Disable for testing
                "max_features": 10,
                "explanation_timeout": 10
            },
            "lime_analysis": {
                "enable_lime": False,  # Disable for testing
                "max_features": 10,
                "num_samples": 100,
                "explanation_timeout": 10
            },
            "trading_integration": {
                "enable_monitoring": True,
                "capture_explanations": True,
                "capture_performance_metrics": True,
                "real_time_export": False,
                "export_interval_minutes": 1,
                "max_memory_decisions": 1000
            },
            "trade_decision_capture": {
                "enable_market_conditions": True,
                "enable_hmm_context": True,
                "enable_signal_context": True,
                "enable_model_context": True,
                "enable_ensemble_context": True
            }
        }
    
    async def test_initialization(self):
        """Test system initialization."""
        print("🧪 Testing system initialization...")
        
        try:
            # Initialize orchestrator
            self.orchestrator = EnhancedMonitoringOrchestrator(self.config)
            assert self.orchestrator is not None
            print("✅ Orchestrator initialized")
            
            # Initialize context capture
            self.context_capture = TradeDecisionContextCapture(self.config)
            assert self.context_capture is not None
            print("✅ Context capture initialized")
            
            # Initialize explainability integrator
            self.explainability_integrator = ExplainabilityIntegrator(self.config)
            assert self.explainability_integrator is not None
            print("✅ Explainability integrator initialized")
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def test_context_capture(self):
        """Test trade decision context capture."""
        print("🧪 Testing context capture...")
        
        try:
            # Create mock market data
            market_data = pd.DataFrame({
                'close': [100, 101, 102, 103, 104],
                'volume': [1000, 1100, 1200, 1300, 1400],
                'high': [101, 102, 103, 104, 105],
                'low': [99, 100, 101, 102, 103],
                'rsi_14': [50, 55, 60, 65, 70],
                'macd': [0.1, 0.2, 0.3, 0.4, 0.5],
                'bb_position': [0.3, 0.4, 0.5, 0.6, 0.7],
                'atr_14': [1.0, 1.1, 1.2, 1.3, 1.4],
                'adx_14': [25, 30, 35, 40, 45]
            })
            
            # Test context capture
            context = await self.context_capture.capture_trade_context(
                exchange="test_exchange",
                symbol="TESTUSDT",
                trading_mode=TradingMode.BACKTEST,
                current_price=104.0,
                current_volume=1400.0,
                price_history=market_data['close'].tolist(),
                volume_history=market_data['volume'].tolist(),
                market_data=market_data
            )
            
            assert context is not None
            assert context.exchange == "test_exchange"
            assert context.symbol == "TESTUSDT"
            assert context.current_price == 104.0
            print("✅ Context capture successful")
            
            return True
            
        except Exception as e:
            print(f"❌ Context capture failed: {e}")
            return False
    
    async def test_decision_recording(self):
        """Test comprehensive decision recording."""
        print("🧪 Testing decision recording...")
        
        try:
            # Create test data
            trade_context = TradeContext(
                exchange="test_exchange",
                token="TESTUSDT",
                timestamp=datetime.now(),
                price=104.0,
                volume=1400.0,
                timeframe="15m"
            )
            
            trading_indicators = [
                TradingIndicator(
                    name="RSI",
                    value=70.0,
                    weight=0.3,
                    confidence=0.8,
                    risk_score=0.2,
                    description="RSI indicator"
                ),
                TradingIndicator(
                    name="MACD",
                    value=0.5,
                    weight=0.4,
                    confidence=0.7,
                    risk_score=0.3,
                    description="MACD indicator"
                )
            ]
            
            model_decisions = [
                MLModelDecision(
                    model_id="test_model_1",
                    model_type=ModelType.HMM,
                    prediction=0.7,
                    confidence=0.8,
                    risk_score=0.2,
                    feature_importance={"price": 0.6, "volume": 0.4},
                    processing_time_ms=10.0,
                    model_version="1.0"
                ),
                MLModelDecision(
                    model_id="test_model_2",
                    model_type=ModelType.ANALYST,
                    prediction=0.6,
                    confidence=0.75,
                    risk_score=0.25,
                    feature_importance={"price": 0.5, "volume": 0.5},
                    processing_time_ms=15.0,
                    model_version="1.0"
                )
            ]
            
            ensemble_decision = EnsembleDecision(
                ensemble_id="test_ensemble",
                final_prediction=0.65,
                final_confidence=0.775,
                final_risk_score=0.225,
                model_weights={"test_model_1": 0.6, "test_model_2": 0.4},
                model_decisions=model_decisions,
                voting_mechanism="weighted_average",
                consensus_score=0.9,
                disagreement_level=0.1
            )
            
            model_indicator_weights = {
                "test_model_1": {"RSI": 0.3, "MACD": 0.7},
                "test_model_2": {"RSI": 0.4, "MACD": 0.6}
            }
            
            # Record decision
            decision = await self.orchestrator.record_comprehensive_decision(
                context=trade_context,
                trading_mode=TradingMode.BACKTEST,
                trading_indicators=trading_indicators,
                ensemble_decision=ensemble_decision,
                individual_model_decisions=model_decisions,
                model_indicator_weights=model_indicator_weights,
                action="buy",
                position_size=0.1,
                stop_loss=100.0,
                take_profit=110.0
            )
            
            assert decision is not None
            assert decision.action == "buy"
            assert decision.position_size == 0.1
            assert len(decision.individual_model_decisions) == 2
            print("✅ Decision recording successful")
            
            return True
            
        except Exception as e:
            print(f"❌ Decision recording failed: {e}")
            return False
    
    async def test_export_functionality(self):
        """Test export functionality."""
        print("🧪 Testing export functionality...")
        
        try:
            # Test daily ongoing CSV export
            daily_success = await self.orchestrator.export_daily_ongoing_csv()
            assert daily_success
            print("✅ Daily ongoing CSV export successful")
            
            # Test monthly report export
            monthly_success = await self.orchestrator.export_monthly_report()
            assert monthly_success
            print("✅ Monthly report export successful")
            
            # Test force export all
            export_success = await self.orchestrator.force_export_all()
            assert export_success
            print("✅ Force export all successful")
            
            # Check if files were created
            export_dir = Path(self.config["enhanced_monitoring"]["export_directory"])
            assert export_dir.exists()
            
            # Check for ongoing CSV
            ongoing_csv = export_dir / "ongoing_daily_metrics.csv"
            if ongoing_csv.exists():
                print("✅ Ongoing daily metrics CSV created")
            
            # Check for monthly report directory
            monthly_dirs = list(export_dir.glob("monthly_reports_*"))
            if monthly_dirs:
                print("✅ Monthly report directory created")
            
            return True
            
        except Exception as e:
            print(f"❌ Export functionality failed: {e}")
            return False
    
    async def test_statistics(self):
        """Test statistics functionality."""
        print("🧪 Testing statistics...")
        
        try:
            stats = self.orchestrator.get_monitoring_stats()
            assert stats is not None
            assert "orchestrator_stats" in stats
            assert "enhanced_ml_monitor_stats" in stats
            
            print(f"✅ Statistics retrieved: {len(stats)} categories")
            
            # Test context capture stats
            context_stats = self.context_capture.get_capture_stats()
            assert context_stats is not None
            print(f"✅ Context capture stats: {context_stats['total_contexts_captured']} contexts")
            
            # Test explainability stats
            explainability_stats = self.explainability_integrator.get_explanation_stats()
            assert explainability_stats is not None
            print(f"✅ Explainability stats: {len(explainability_stats)} metrics")
            
            return True
            
        except Exception as e:
            print(f"❌ Statistics test failed: {e}")
            return False
    
    async def test_cleanup(self):
        """Test cleanup functionality."""
        print("🧪 Testing cleanup...")
        
        try:
            # Test cleanup
            await self.orchestrator._cleanup_old_data()
            print("✅ Cleanup successful")
            
            return True
            
        except Exception as e:
            print(f"❌ Cleanup test failed: {e}")
            return False
    
    def teardown(self):
        """Cleanup test environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print("🧹 Test environment cleaned up")
    
    async def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting Enhanced Monitoring System Tests")
        print("=" * 50)
        
        self.setup()
        
        try:
            tests = [
                ("Initialization", self.test_initialization),
                ("Context Capture", self.test_context_capture),
                ("Decision Recording", self.test_decision_recording),
                ("Export Functionality", self.test_export_functionality),
                ("Statistics", self.test_statistics),
                ("Cleanup", self.test_cleanup)
            ]
            
            passed = 0
            total = len(tests)
            
            for test_name, test_func in tests:
                print(f"\n📋 Running {test_name} test...")
                try:
                    success = await test_func()
                    if success:
                        passed += 1
                        print(f"✅ {test_name} test passed")
                    else:
                        print(f"❌ {test_name} test failed")
                except Exception as e:
                    print(f"❌ {test_name} test failed with exception: {e}")
            
            print("\n" + "=" * 50)
            print(f"📊 Test Results: {passed}/{total} tests passed")
            
            if passed == total:
                print("🎉 All tests passed! Enhanced monitoring system is working correctly.")
                return True
            else:
                print("⚠️ Some tests failed. Please check the implementation.")
                return False
                
        finally:
            self.teardown()


async def main():
    """Main test function."""
    tester = TestEnhancedMonitoring()
    success = await tester.run_all_tests()
    
    if success:
        print("\n✅ Enhanced Monitoring System is ready for use!")
    else:
        print("\n❌ Enhanced Monitoring System has issues that need to be addressed.")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)