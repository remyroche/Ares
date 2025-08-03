#!/usr/bin/env python3
"""
Enhanced Trade Tracking Demo

This script demonstrates the comprehensive trade tracking system with:
- Model ensemble tracking
- Regime analysis
- Feature importance tracking
- Decision path analysis
- Model behavior monitoring
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.supervisor.performance_monitor import setup_performance_monitor
from src.supervisor.model_behavior_tracker import setup_model_behavior_tracker
from src.tracking.trade_tracker import TradeTracker


class EnhancedTradeTrackingDemo:
    """
    Demo class for enhanced trade tracking system.
    """
    
    def __init__(self):
        """Initialize the demo."""
        self.logger = system_logger.getChild("EnhancedTradeTrackingDemo")
        
        # Configuration
        self.config = {
            "performance_monitor": {
                "monitor_interval": 30,
                "max_history": 100,
                "concept_drift": {
                    "detection_window": 100,
                    "drift_threshold": 0.05,
                }
            },
            "model_behavior_tracker": {
                "tracking_interval": 60,
                "max_history_size": 1000,
            },
            "trade_tracking": {
                "enable_feature_importance_tracking": True,
                "enable_decision_path_tracking": True,
                "enable_model_behavior_tracking": True,
                "max_history_size": 10000,
            }
        }
        
        # Components
        self.performance_monitor = None
        self.behavior_tracker = None
        self.trade_tracker = None
        
        self.logger.info("🚀 Enhanced Trade Tracking Demo initialized")

    @asyncio.coroutine
    async def setup(self) -> bool:
        """Set up all tracking components."""
        try:
            self.logger.info("Setting up enhanced trade tracking components...")
            
            # Set up performance monitor
            self.performance_monitor = await setup_performance_monitor(self.config)
            if not self.performance_monitor:
                self.logger.error("Failed to set up performance monitor")
                return False
            
            # Set up behavior tracker
            self.behavior_tracker = await setup_model_behavior_tracker(self.config, self.performance_monitor)
            if not self.behavior_tracker:
                self.logger.error("Failed to set up behavior tracker")
                return False
            
            # Set up trade tracker
            self.trade_tracker = TradeTracker(self.config)
            
            self.logger.info("✅ All tracking components set up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to set up tracking components: {e}")
            return False

    async def start_monitoring(self) -> bool:
        """Start all monitoring components."""
        try:
            self.logger.info("Starting monitoring components...")
            
            # Start performance monitor
            await self.performance_monitor.run()
            
            # Start behavior tracker
            await self.behavior_tracker.start_tracking()
            
            self.logger.info("✅ All monitoring components started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start monitoring: {e}")
            return False

    def generate_sample_trade_data(self) -> Dict[str, Any]:
        """Generate sample trade data for demonstration."""
        trade_id = f"trade_{int(time.time() * 1000)}"
        
        # Sample ensemble decision
        ensemble_decision = {
            "ensemble_id": "ensemble_001",
            "ensemble_type": "multi_timeframe_ensemble",
            "primary_prediction": "buy",
            "primary_confidence": 0.85,
            "individual_predictions": [
                {
                    "model_type": "xgboost",
                    "model_id": "xgboost_001",
                    "prediction": "buy",
                    "confidence": 0.82,
                    "probability": {"buy": 0.82, "sell": 0.12, "hold": 0.06},
                    "features_used": ["rsi", "macd", "bollinger_bands", "volume"],
                    "feature_importance": [
                        {
                            "feature_name": "rsi",
                            "importance_score": 0.25,
                            "importance_rank": 1,
                            "model_type": "xgboost",
                            "timeframe": "1h",
                            "regime": "trending"
                        },
                        {
                            "feature_name": "macd",
                            "importance_score": 0.20,
                            "importance_rank": 2,
                            "model_type": "xgboost",
                            "timeframe": "1h",
                            "regime": "trending"
                        }
                    ],
                    "prediction_time": datetime.now().isoformat(),
                    "model_version": "v1.2.3"
                },
                {
                    "model_type": "lstm",
                    "model_id": "lstm_001",
                    "prediction": "buy",
                    "confidence": 0.88,
                    "probability": {"buy": 0.88, "sell": 0.08, "hold": 0.04},
                    "features_used": ["price_sequence", "volume_sequence", "technical_indicators"],
                    "feature_importance": [
                        {
                            "feature_name": "price_sequence",
                            "importance_score": 0.35,
                            "importance_rank": 1,
                            "model_type": "lstm",
                            "timeframe": "1h",
                            "regime": "trending"
                        }
                    ],
                    "prediction_time": datetime.now().isoformat(),
                    "model_version": "v1.1.5"
                }
            ],
            "ensemble_weights": {"xgboost": 0.4, "lstm": 0.6},
            "meta_learner_prediction": "buy",
            "meta_learner_confidence": 0.87
        }
        
        # Sample regime analysis
        regime_analysis = {
            "regime_type": "trending",
            "regime_confidence": 0.78,
            "regime_probabilities": {
                "trending": 0.78,
                "ranging": 0.15,
                "volatile": 0.07
            },
            "regime_features": ["trend_strength", "volatility", "momentum"],
            "regime_indicators": {
                "trend_strength": 0.75,
                "volatility": 0.45,
                "momentum": 0.82
            },
            "regime_transition_probability": 0.12,
            "regime_duration": 45
        }
        
        # Sample decision path
        decision_path = {
            "decision_steps": [
                "feature_extraction",
                "regime_classification",
                "model_prediction",
                "ensemble_combination",
                "risk_assessment",
                "final_decision"
            ],
            "decision_reasons": [
                "Strong technical indicators",
                "Trending market regime",
                "High model confidence",
                "Ensemble agreement",
                "Acceptable risk level",
                "Buy signal confirmed"
            ],
            "decision_weights": [0.1, 0.2, 0.3, 0.2, 0.1, 0.1],
            "decision_thresholds": {
                "confidence_threshold": 0.75,
                "risk_threshold": 0.25,
                "agreement_threshold": 0.8
            },
            "decision_metadata": {
                "processing_time": 0.045,
                "data_quality": "high",
                "model_versions": ["v1.2.3", "v1.1.5"]
            }
        }
        
        # Sample model behaviors
        model_behaviors = [
            {
                "model_type": "xgboost",
                "prediction_consistency": 0.82,
                "confidence_trend": [0.80, 0.82, 0.85, 0.83, 0.84],
                "feature_importance_stability": 0.78,
                "prediction_drift": 0.03,
                "model_performance_metrics": {
                    "accuracy": 0.82,
                    "precision": 0.79,
                    "recall": 0.85,
                    "f1_score": 0.82
                },
                "last_retraining": datetime.now().isoformat()
            },
            {
                "model_type": "lstm",
                "prediction_consistency": 0.88,
                "confidence_trend": [0.85, 0.87, 0.88, 0.89, 0.88],
                "feature_importance_stability": 0.85,
                "prediction_drift": 0.02,
                "model_performance_metrics": {
                    "accuracy": 0.88,
                    "precision": 0.86,
                    "recall": 0.90,
                    "f1_score": 0.88
                },
                "last_retraining": datetime.now().isoformat()
            }
        ]
        
        # Sample trade data
        trade_data = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 0.1,
            "price": 45000.0,
            "timestamp": datetime.now().isoformat(),
            "status": "open",
            "order_type": "market",
            "market_conditions": {
                "volatility": 0.25,
                "volume": 1250000,
                "spread": 0.05
            },
            "risk_metrics": {
                "var_95": 0.02,
                "max_drawdown": 0.05,
                "sharpe_ratio": 1.2
            },
            "execution_metadata": {
                "execution_time": 0.045,
                "slippage": 0.001,
                "fill_quality": "excellent"
            },
            "stop_loss": 44000.0,
            "take_profit": 47000.0
        }
        
        return {
            "trade_data": trade_data,
            "ensemble_decision": ensemble_decision,
            "regime_analysis": regime_analysis,
            "decision_path": decision_path,
            "model_behaviors": model_behaviors
        }

    async def demonstrate_trade_tracking(self) -> None:
        """Demonstrate comprehensive trade tracking."""
        try:
            self.logger.info("🎯 Demonstrating enhanced trade tracking...")
            
            # Generate sample trade data
            sample_data = self.generate_sample_trade_data()
            
            # Record trade with comprehensive tracking
            success = await self.trade_tracker.record_trade(
                trade_data=sample_data["trade_data"],
                ensemble_decision=sample_data["ensemble_decision"],
                regime_analysis=sample_data["regime_analysis"],
                decision_path=sample_data["decision_path"],
                model_behaviors=sample_data["model_behaviors"]
            )
            
            if success:
                self.logger.info("✅ Trade recorded successfully")
                
                # Demonstrate analysis capabilities
                await self.demonstrate_analysis_capabilities()
            else:
                self.logger.error("❌ Failed to record trade")
                
        except Exception as e:
            self.logger.error(f"Error demonstrating trade tracking: {e}")

    async def demonstrate_analysis_capabilities(self) -> None:
        """Demonstrate various analysis capabilities."""
        try:
            self.logger.info("📊 Demonstrating analysis capabilities...")
            
            # Get trade history
            trade_history = self.trade_tracker.get_trade_history(limit=10)
            self.logger.info(f"📈 Retrieved {len(trade_history)} trades from history")
            
            # Get performance metrics
            performance_metrics = self.trade_tracker.get_performance_metrics()
            self.logger.info(f"📊 Performance metrics: {performance_metrics}")
            
            # Get model performance summary
            model_performance = self.trade_tracker.get_model_performance_summary()
            self.logger.info(f"🎯 Model performance summary: {model_performance}")
            
            # Get feature importance analysis
            feature_analysis = self.trade_tracker.get_feature_importance_analysis()
            self.logger.info(f"📈 Feature importance analysis: {feature_analysis}")
            
            # Get decision path analysis
            decision_analysis = self.trade_tracker.get_decision_path_analysis()
            self.logger.info(f"🛤️ Decision path analysis: {decision_analysis}")
            
            # Get regime analysis summary
            regime_analysis = self.trade_tracker.get_regime_analysis_summary()
            self.logger.info(f"📊 Regime analysis summary: {regime_analysis}")
            
            # Get behavior summaries from behavior tracker
            behavior_summaries = self.behavior_tracker.get_all_behavior_summaries()
            self.logger.info(f"🧠 Behavior summaries: {behavior_summaries}")
            
        except Exception as e:
            self.logger.error(f"Error demonstrating analysis capabilities: {e}")

    async def demonstrate_continuous_monitoring(self, duration_minutes: int = 5) -> None:
        """Demonstrate continuous monitoring over time."""
        try:
            self.logger.info(f"⏱️ Demonstrating continuous monitoring for {duration_minutes} minutes...")
            
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            while datetime.now() < end_time:
                # Generate and record a new trade every 30 seconds
                sample_data = self.generate_sample_trade_data()
                
                await self.trade_tracker.record_trade(
                    trade_data=sample_data["trade_data"],
                    ensemble_decision=sample_data["ensemble_decision"],
                    regime_analysis=sample_data["regime_analysis"],
                    decision_path=sample_data["decision_path"],
                    model_behaviors=sample_data["model_behaviors"]
                )
                
                # Wait 30 seconds before next trade
                await asyncio.sleep(30)
            
            self.logger.info("✅ Continuous monitoring demonstration completed")
            
        except Exception as e:
            self.logger.error(f"Error in continuous monitoring: {e}")

    async def export_demonstration_data(self) -> None:
        """Export demonstration data for analysis."""
        try:
            self.logger.info("📤 Exporting demonstration data...")
            
            # Export trade data
            trade_export_path = self.trade_tracker.export_trade_data(format="json")
            self.logger.info(f"📊 Trade data exported to: {trade_export_path}")
            
            # Export behavior data
            behavior_export_path = self.behavior_tracker.export_behavior_data()
            self.logger.info(f"🧠 Behavior data exported to: {behavior_export_path}")
            
            # Create summary report
            await self.create_summary_report()
            
        except Exception as e:
            self.logger.error(f"Error exporting demonstration data: {e}")

    async def create_summary_report(self) -> None:
        """Create a comprehensive summary report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"enhanced_tracking_demo_report_{timestamp}.json"
            
            report = {
                "demo_timestamp": datetime.now().isoformat(),
                "demo_duration": "5 minutes",
                "components_tested": [
                    "Trade Tracker",
                    "Model Behavior Tracker",
                    "Performance Monitor"
                ],
                "trade_tracking_summary": {
                    "total_trades": len(self.trade_tracker.trade_history),
                    "performance_metrics": self.trade_tracker.get_performance_metrics(),
                    "model_performance": self.trade_tracker.get_model_performance_summary(),
                    "feature_analysis": self.trade_tracker.get_feature_importance_analysis(),
                    "decision_analysis": self.trade_tracker.get_decision_path_analysis(),
                    "regime_analysis": self.trade_tracker.get_regime_analysis_summary()
                },
                "behavior_tracking_summary": {
                    "models_tracked": len(self.behavior_tracker.behavior_history),
                    "behavior_summaries": self.behavior_tracker.get_all_behavior_summaries()
                },
                "performance_monitoring_summary": {
                    "status": self.performance_monitor.get_status(),
                    "metrics": self.performance_monitor.get_performance_metrics(),
                    "alerts": self.performance_monitor.get_alerts()
                },
                "capabilities_demonstrated": [
                    "Model ensemble tracking with individual model predictions",
                    "Regime analysis with confidence and transition probabilities",
                    "Feature importance tracking with stability metrics",
                    "Decision path analysis with step-by-step reasoning",
                    "Model behavior monitoring with drift detection",
                    "Continuous performance monitoring",
                    "Comprehensive data export and analysis"
                ]
            }
            
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"📋 Summary report created: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating summary report: {e}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.logger.info("🧹 Cleaning up resources...")
            
            # Stop monitoring components
            if self.behavior_tracker:
                await self.behavior_tracker.stop_tracking()
            
            if self.performance_monitor:
                await self.performance_monitor.stop()
            
            self.logger.info("✅ Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


async def main():
    """Main demonstration function."""
    demo = EnhancedTradeTrackingDemo()
    
    try:
        # Set up components
        if not await demo.setup():
            return
        
        # Start monitoring
        if not await demo.start_monitoring():
            return
        
        # Demonstrate trade tracking
        await demo.demonstrate_trade_tracking()
        
        # Demonstrate continuous monitoring
        await demo.demonstrate_continuous_monitoring(duration_minutes=2)
        
        # Export demonstration data
        await demo.export_demonstration_data()
        
        # Clean up
        await demo.cleanup()
        
        print("\n🎉 Enhanced Trade Tracking Demo completed successfully!")
        print("📊 Check the exported files for detailed analysis.")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
        await demo.cleanup()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main()) 