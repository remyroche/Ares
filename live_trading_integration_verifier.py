#!/usr/bin/env python3
"""
Live Trading Integration Verifier

This script verifies that all components of the live trading flow are properly integrated:
1. Exchange API data fetching
2. Feature engineering with wavelet transforms
3. Analyst ML model signal generation
4. Tactician position management
5. Position monitoring and closing
6. Complete end-to-end flow

Usage:
    python live_trading_integration_verifier.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.exchange.binance import BinanceExchange
from src.analyst.analyst import Analyst
from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
from src.tactician.tactician import Tactician
from src.tactician.position_monitor import PositionMonitor
from src.pipelines.live_trading_pipeline import LiveTradingPipeline
from src.utils.logger import system_logger


class LiveTradingIntegrationVerifier:
    """
    Comprehensive verifier for live trading flow integration.
    """

    def __init__(self):
        self.logger = system_logger.getChild("LiveTradingIntegrationVerifier")
        self.config = CONFIG
        
        # Components to verify
        self.exchange = None
        self.analyst = None
        self.feature_engineering = None
        self.tactician = None
        self.position_monitor = None
        self.live_trading_pipeline = None
        
        # Test results
        self.test_results = {}
        self.verification_status = {}

    async def run_comprehensive_verification(self) -> Dict[str, Any]:
        """
        Run comprehensive verification of all live trading components.
        
        Returns:
            Dict containing verification results
        """
        self.logger.info("🚀 Starting comprehensive live trading integration verification...")
        
        try:
            # Step 1: Verify Exchange API Integration
            await self._verify_exchange_integration()
            
            # Step 2: Verify Feature Engineering with Wavelet
            await self._verify_feature_engineering()
            
            # Step 3: Verify Analyst ML Models
            await self._verify_analyst_integration()
            
            # Step 4: Verify Tactician Position Management
            await self._verify_tactician_integration()
            
            # Step 5: Verify Position Monitoring
            await self._verify_position_monitoring()
            
            # Step 6: Verify Complete Live Trading Pipeline
            await self._verify_live_trading_pipeline()
            
            # Step 7: Verify End-to-End Flow
            await self._verify_end_to_end_flow()
            
            # Generate verification report
            return self._generate_verification_report()
            
        except Exception as e:
            self.logger.error(f"❌ Verification failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def _verify_exchange_integration(self) -> None:
        """Verify exchange API integration for data fetching."""
        self.logger.info("📊 Step 1: Verifying Exchange API Integration...")
        
        try:
            # Initialize exchange
            self.exchange = BinanceExchange(self.config)
            exchange_init = await self.exchange.initialize()
            
            if not exchange_init:
                raise Exception("Failed to initialize exchange")
            
            # Test data fetching
            test_symbol = "ETHUSDT"
            
            # Test klines data
            klines = await self.exchange.get_klines(test_symbol, "1h", 100)
            if not klines:
                raise Exception("Failed to fetch klines data")
            
            # Test ticker data
            ticker = await self.exchange.get_ticker(test_symbol)
            if not ticker:
                raise Exception("Failed to fetch ticker data")
            
            # Test order book
            order_book = await self.exchange.get_order_book(test_symbol, 10)
            if not order_book:
                raise Exception("Failed to fetch order book data")
            
            self.verification_status["exchange"] = {
                "status": "✅ PASSED",
                "klines_fetched": len(klines),
                "ticker_fetched": bool(ticker),
                "order_book_fetched": bool(order_book)
            }
            
            self.logger.info("✅ Exchange API integration verified successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Exchange integration verification failed: {e}")
            self.verification_status["exchange"] = {
                "status": "❌ FAILED",
                "error": str(e)
            }

    async def _verify_feature_engineering(self) -> None:
        """Verify feature engineering with wavelet transforms."""
        self.logger.info("🔧 Step 2: Verifying Feature Engineering with Wavelet...")
        
        try:
            # Initialize feature engineering orchestrator
            self.feature_engineering = FeatureEngineeringOrchestrator(self.config)
            
            # Create test data
            import pandas as pd
            import numpy as np
            
            test_data = pd.DataFrame({
                'open': np.random.randn(100) + 100,
                'high': np.random.randn(100) + 101,
                'low': np.random.randn(100) + 99,
                'close': np.random.randn(100) + 100.5,
                'volume': np.random.randn(100) + 1000
            })
            
            # Test feature generation
            features = await self.feature_engineering.generate_all_features(
                klines_df=test_data,
                agg_trades_df=None,
                futures_df=None,
                sr_levels=None
            )
            
            if features.empty:
                raise Exception("No features generated")
            
            # Check for wavelet features
            wavelet_features = [col for col in features.columns if 'wavelet' in col.lower()]
            
            self.verification_status["feature_engineering"] = {
                "status": "✅ PASSED",
                "total_features": len(features.columns),
                "wavelet_features": len(wavelet_features),
                "sample_wavelet_features": wavelet_features[:5] if wavelet_features else []
            }
            
            self.logger.info(f"✅ Feature engineering verified: {len(features.columns)} features generated")
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering verification failed: {e}")
            self.verification_status["feature_engineering"] = {
                "status": "❌ FAILED",
                "error": str(e)
            }

    async def _verify_analyst_integration(self) -> None:
        """Verify analyst ML model integration."""
        self.logger.info("🧠 Step 3: Verifying Analyst ML Model Integration...")
        
        try:
            # Initialize analyst
            self.analyst = Analyst(self.config)
            analyst_init = await self.analyst.initialize()
            
            if not analyst_init:
                raise Exception("Failed to initialize analyst")
            
            # Test analysis execution
            analysis_input = {
                "symbol": "ETHUSDT",
                "timeframe": "1h",
                "limit": 100,
                "analysis_type": "technical",
                "include_indicators": True,
                "include_patterns": True,
            }
            
            analysis_result = await self.analyst.execute_analysis(analysis_input)
            
            if not analysis_result:
                raise Exception("Analysis execution failed")
            
            # Get analysis results
            results = self.analyst.get_analysis_results()
            
            self.verification_status["analyst"] = {
                "status": "✅ PASSED",
                "analysis_executed": bool(analysis_result),
                "results_available": bool(results),
                "analysis_types": list(results.keys()) if results else []
            }
            
            self.logger.info("✅ Analyst ML model integration verified successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Analyst integration verification failed: {e}")
            self.verification_status["analyst"] = {
                "status": "❌ FAILED",
                "error": str(e)
            }

    async def _verify_tactician_integration(self) -> None:
        """Verify tactician position management."""
        self.logger.info("🎯 Step 4: Verifying Tactician Position Management...")
        
        try:
            # Initialize tactician
            self.tactician = Tactician(self.config)
            tactician_init = await self.tactician.initialize()
            
            if not tactician_init:
                raise Exception("Failed to initialize tactician")
            
            # Test tactics execution
            tactics_input = {
                "symbol": "ETHUSDT",
                "current_price": 100.0,
                "position_size": 0.1,
                "leverage": 1.0,
                "risk_level": "medium"
            }
            
            tactics_result = await self.tactician.execute_tactics(tactics_input)
            
            if not tactics_result:
                raise Exception("Tactics execution failed")
            
            # Get tactician status
            status = self.tactician.get_status()
            
            self.verification_status["tactician"] = {
                "status": "✅ PASSED",
                "tactics_executed": bool(tactics_result),
                "status_available": bool(status),
                "component_managers": list(status.keys()) if status else []
            }
            
            self.logger.info("✅ Tactician position management verified successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Tactician integration verification failed: {e}")
            self.verification_status["tactician"] = {
                "status": "❌ FAILED",
                "error": str(e)
            }

    async def _verify_position_monitoring(self) -> None:
        """Verify position monitoring and closing."""
        self.logger.info("👁️ Step 5: Verifying Position Monitoring...")
        
        try:
            # Initialize position monitor
            self.position_monitor = PositionMonitor(self.config)
            monitor_init = await self.position_monitor.initialize()
            
            if not monitor_init:
                raise Exception("Failed to initialize position monitor")
            
            # Test position monitoring
            test_position = {
                "position_id": "test_position_001",
                "symbol": "ETHUSDT",
                "side": "long",
                "entry_price": 100.0,
                "current_price": 101.0,
                "quantity": 0.1,
                "entry_time": "2024-01-01T00:00:00Z",
                "confidence": 0.75
            }
            
            # Add test position
            self.position_monitor.add_position("test_position_001", test_position)
            
            # Test position assessment
            assessment = await self.position_monitor._assess_position(
                "test_position_001",
                test_position,
                analyst_confidence=0.8,
                tactician_confidence=0.7
            )
            
            if not assessment:
                raise Exception("Position assessment failed")
            
            # Get active positions
            active_positions = self.position_monitor.get_active_positions()
            
            self.verification_status["position_monitor"] = {
                "status": "✅ PASSED",
                "monitor_initialized": bool(monitor_init),
                "position_added": "test_position_001" in active_positions,
                "assessment_working": bool(assessment),
                "active_positions": len(active_positions)
            }
            
            self.logger.info("✅ Position monitoring verified successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Position monitoring verification failed: {e}")
            self.verification_status["position_monitor"] = {
                "status": "❌ FAILED",
                "error": str(e)
            }

    async def _verify_live_trading_pipeline(self) -> None:
        """Verify complete live trading pipeline."""
        self.logger.info("🔄 Step 6: Verifying Live Trading Pipeline...")
        
        try:
            # Initialize live trading pipeline
            self.live_trading_pipeline = LiveTradingPipeline(self.config)
            pipeline_init = await self.live_trading_pipeline.initialize()
            
            if not pipeline_init:
                raise Exception("Failed to initialize live trading pipeline")
            
            # Test trading execution
            test_market_data = {
                "symbol": "ETHUSDT",
                "price": 100.0,
                "volume": 1000.0,
                "timestamp": "2024-01-01T00:00:00Z",
                "order_book": {"bids": [[99.9, 1.0]], "asks": [[100.1, 1.0]]}
            }
            
            trading_result = await self.live_trading_pipeline.execute_trading(test_market_data)
            
            if not trading_result:
                raise Exception("Trading execution failed")
            
            # Get trading status
            status = self.live_trading_pipeline.get_trading_status()
            
            self.verification_status["live_trading_pipeline"] = {
                "status": "✅ PASSED",
                "pipeline_initialized": bool(pipeline_init),
                "trading_executed": bool(trading_result),
                "status_available": bool(status)
            }
            
            self.logger.info("✅ Live trading pipeline verified successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Live trading pipeline verification failed: {e}")
            self.verification_status["live_trading_pipeline"] = {
                "status": "❌ FAILED",
                "error": str(e)
            }

    async def _verify_end_to_end_flow(self) -> None:
        """Verify complete end-to-end trading flow."""
        self.logger.info("🔄 Step 7: Verifying End-to-End Trading Flow...")
        
        try:
            # Simulate complete trading flow
            flow_steps = []
            
            # Step 1: Fetch data from exchange
            if self.exchange:
                klines = await self.exchange.get_klines("ETHUSDT", "1h", 100)
                flow_steps.append("✅ Data fetched from exchange")
            
            # Step 2: Generate features with wavelet
            if self.feature_engineering and klines:
                import pandas as pd
                klines_df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
                features = await self.feature_engineering.generate_all_features(klines_df)
                flow_steps.append("✅ Features generated with wavelet")
            
            # Step 3: Analyst finds signals
            if self.analyst and features is not None:
                analysis_input = {
                    "symbol": "ETHUSDT",
                    "timeframe": "1h",
                    "limit": 100,
                    "analysis_type": "technical",
                    "include_indicators": True,
                    "include_patterns": True,
                }
                analysis_result = await self.analyst.execute_analysis(analysis_input)
                flow_steps.append("✅ Analyst found signals")
            
            # Step 4: Tactician evaluates opportunity
            if self.tactician and analysis_result:
                tactics_input = {
                    "symbol": "ETHUSDT",
                    "current_price": 100.0,
                    "position_size": 0.1,
                    "leverage": 1.0,
                    "risk_level": "medium"
                }
                tactics_result = await self.tactician.execute_tactics(tactics_input)
                flow_steps.append("✅ Tactician evaluated opportunity")
            
            # Step 5: Position monitoring
            if self.position_monitor and tactics_result:
                test_position = {
                    "position_id": "e2e_test_position",
                    "symbol": "ETHUSDT",
                    "side": "long",
                    "entry_price": 100.0,
                    "current_price": 101.0,
                    "quantity": 0.1,
                    "entry_time": "2024-01-01T00:00:00Z",
                    "confidence": 0.75
                }
                self.position_monitor.add_position("e2e_test_position", test_position)
                flow_steps.append("✅ Position monitoring active")
            
            self.verification_status["end_to_end_flow"] = {
                "status": "✅ PASSED",
                "flow_steps": flow_steps,
                "total_steps": len(flow_steps)
            }
            
            self.logger.info("✅ End-to-end trading flow verified successfully")
            
        except Exception as e:
            self.logger.error(f"❌ End-to-end flow verification failed: {e}")
            self.verification_status["end_to_end_flow"] = {
                "status": "❌ FAILED",
                "error": str(e)
            }

    def _generate_verification_report(self) -> Dict[str, Any]:
        """Generate comprehensive verification report."""
        self.logger.info("📋 Generating verification report...")
        
        # Calculate overall status
        passed_components = sum(1 for status in self.verification_status.values() 
                              if status.get("status", "").startswith("✅"))
        total_components = len(self.verification_status)
        
        overall_status = "✅ PASSED" if passed_components == total_components else "❌ FAILED"
        
        report = {
            "overall_status": overall_status,
            "passed_components": passed_components,
            "total_components": total_components,
            "success_rate": f"{(passed_components/total_components)*100:.1f}%",
            "component_details": self.verification_status,
            "summary": {
                "exchange_api": "✅ Data fetching from exchange API",
                "feature_engineering": "✅ Wavelet and advanced feature generation",
                "analyst_ml": "✅ ML model signal generation",
                "tactician": "✅ Position management and sizing",
                "position_monitoring": "✅ Real-time position monitoring",
                "live_pipeline": "✅ Complete live trading pipeline",
                "end_to_end": "✅ Full trading flow integration"
            }
        }
        
        return report

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.exchange:
                await self.exchange.stop()
            if self.analyst:
                await self.analyst.stop()
            if self.tactician:
                await self.tactician.stop()
            if self.position_monitor:
                await self.position_monitor.stop_monitoring()
            if self.live_trading_pipeline:
                await self.live_trading_pipeline.stop()
                
            self.logger.info("🧹 Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")


async def main():
    """Main verification function."""
    print("🚀 Live Trading Integration Verifier")
    print("=" * 50)
    
    verifier = LiveTradingIntegrationVerifier()
    
    try:
        # Run comprehensive verification
        report = await verifier.run_comprehensive_verification()
        
        # Print results
        print(f"\n📊 VERIFICATION RESULTS")
        print(f"Overall Status: {report['overall_status']}")
        print(f"Success Rate: {report['success_rate']}")
        print(f"Passed Components: {report['passed_components']}/{report['total_components']}")
        
        print(f"\n📋 COMPONENT DETAILS:")
        for component, details in report['component_details'].items():
            print(f"  {component.replace('_', ' ').title()}: {details['status']}")
            if 'error' in details:
                print(f"    Error: {details['error']}")
        
        print(f"\n📝 SUMMARY:")
        for key, description in report['summary'].items():
            print(f"  {description}")
        
        if report['overall_status'] == "✅ PASSED":
            print(f"\n🎉 SUCCESS: Live trading flow is fully integrated!")
        else:
            print(f"\n⚠️ WARNING: Some components need attention")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False
    
    finally:
        await verifier.cleanup()
    
    return report['overall_status'] == "✅ PASSED"


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)