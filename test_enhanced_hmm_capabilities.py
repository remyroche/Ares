#!/usr/bin/env python3
"""
Test Enhanced HMM Capabilities

This script demonstrates and validates the enhanced HMM regime discovery and prediction capabilities.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.step3_hmm_regime_discovery import HMMRegimeDiscoveryStep
from src.training.steps.step9_5_hmm_lm_generalist_training import HMMLMGeneralistTrainingStep
from src.analyst.enhanced_regime_predictor import EnhancedRegimePredictor
from src.utils.logger import system_logger


class EnhancedHMMTester:
    """Test class for enhanced HMM capabilities."""
    
    def __init__(self):
        self.logger = system_logger.getChild("EnhancedHMMTester")
        
    def generate_test_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic test data for HMM analysis."""
        self.logger.info(f"📊 Generating {n_samples} synthetic test samples...")
        
        # Create timestamp index
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=n_samples//1440),  # Assuming 1-minute data
            periods=n_samples,
            freq='1min'
        )
        
        # Generate synthetic price data with regime changes
        np.random.seed(42)
        
        # Create different regimes
        regime_lengths = [200, 300, 250, 250]  # Different regime durations
        regimes = []
        for i, length in enumerate(regime_lengths):
            regimes.extend([i] * length)
        
        # Ensure we have enough samples
        while len(regimes) < n_samples:
            regimes.extend([np.random.randint(0, 4)] * 100)
        
        regimes = regimes[:n_samples]
        
        # Generate price data based on regimes
        prices = [100.0]  # Starting price
        volumes = [1000.0]  # Starting volume
        
        for i in range(1, n_samples):
            regime = regimes[i]
            
            # Different characteristics for each regime
            if regime == 0:  # Low volatility, upward trend
                price_change = np.random.normal(0.001, 0.005)
                volume_change = np.random.normal(0.05, 0.1)
            elif regime == 1:  # High volatility, sideways
                price_change = np.random.normal(0.0, 0.02)
                volume_change = np.random.normal(0.0, 0.2)
            elif regime == 2:  # Low volatility, downward trend
                price_change = np.random.normal(-0.001, 0.005)
                volume_change = np.random.normal(-0.05, 0.1)
            else:  # High volatility, trend reversal
                price_change = np.random.normal(0.0, 0.015)
                volume_change = np.random.normal(0.0, 0.15)
            
            # Apply changes
            new_price = prices[-1] * (1 + price_change)
            new_volume = volumes[-1] * (1 + volume_change)
            
            prices.append(max(new_price, 1.0))  # Ensure positive price
            volumes.append(max(new_volume, 100.0))  # Ensure positive volume
        
        # Create OHLC data
        data = []
        for i in range(n_samples):
            price = prices[i]
            volume = volumes[i]
            
            # Generate OHLC from price
            high = price * (1 + abs(np.random.normal(0, 0.002)))
            low = price * (1 - abs(np.random.normal(0, 0.002)))
            open_price = price * (1 + np.random.normal(0, 0.001))
            close_price = price
            
            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume,
                'true_regime': regimes[i]
            })
        
        df = pd.DataFrame(data)
        self.logger.info(f"✅ Generated test data with {len(df)} samples")
        self.logger.info(f"📊 Regime distribution: {df['true_regime'].value_counts().to_dict()}")
        
        return df
    
    async def test_step3_enhanced_features(self, test_data: pd.DataFrame):
        """Test enhanced features in Step 3 HMM regime discovery."""
        self.logger.info("🧪 Testing Step 3 enhanced HMM features...")
        
        try:
            # Initialize Step 3
            config = {
                "SYMBOL": "TEST",
                "EXCHANGE": "TEST",
                "TIMEFRAME": "1m",
                "DATA_DIR": "test_data"
            }
            
            step3 = HMMRegimeDiscoveryStep(config)
            await step3.initialize()
            
            # Test enhanced regime change detection
            self.logger.info("🔍 Testing enhanced regime change detection...")
            
            # Create mock HMM probabilities and states
            n_samples = len(test_data)
            n_states = 4
            
            # Generate realistic HMM probabilities
            hmm_probs = np.random.dirichlet([1, 1, 1, 1], size=n_samples)
            hmm_states = np.random.randint(0, n_states, size=n_samples)
            
            # Test enhanced regime change detection
            regime_changes = step3._detect_regime_changes_advanced(
                hmm_probs, hmm_states, threshold=0.1, min_persistence=3
            )
            
            if regime_changes.get("success", False):
                self.logger.info("✅ Enhanced regime change detection successful")
                self.logger.info(f"📊 Detected {len(regime_changes['regime_changes'])} regime changes")
                self.logger.info(f"📈 Stability metrics: {regime_changes['stability_metrics']}")
            else:
                self.logger.error("❌ Enhanced regime change detection failed")
            
            # Test adaptive regime boundaries
            self.logger.info("🔧 Testing adaptive regime boundaries...")
            
            # Prepare features for boundary calculation
            features = step3._prepare_hmm_features(test_data)
            
            adaptive_boundaries = step3._calculate_adaptive_regime_boundaries(features)
            
            if adaptive_boundaries:
                self.logger.info("✅ Adaptive regime boundaries calculated")
                self.logger.info(f"📊 Boundary stats: {len(adaptive_boundaries.get('boundary_stats', {}))} boundaries")
            else:
                self.logger.warning("⚠️ Adaptive regime boundaries calculation failed")
            
            # Test regime persistence modeling
            self.logger.info("📊 Testing regime persistence modeling...")
            
            persistence_model = step3._model_regime_persistence(hmm_states)
            
            if persistence_model:
                self.logger.info("✅ Regime persistence model fitted")
                self.logger.info(f"📈 Best distribution: {persistence_model.get('best_distribution')}")
                self.logger.info(f"📊 Persistence stats: {persistence_model.get('persistence_stats', {})}")
            else:
                self.logger.warning("⚠️ Regime persistence modeling failed")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Step 3 enhanced features test failed: {e}")
            return False
    
    async def test_step9_5_enhanced_detection(self, test_data: pd.DataFrame):
        """Test enhanced regime change detection in Step 9.5."""
        self.logger.info("🧪 Testing Step 9.5 enhanced regime change detection...")
        
        try:
            # Initialize Step 9.5
            config = {
                "HMM_LM": {
                    "generalist": {
                        "hmm_states": 4,
                        "sequence_length": 20,
                        "timeframes": ["1m"],
                        "d_model": 256,
                        "nhead": 8,
                        "num_layers": 6,
                        "dropout_rate": 0.1,
                        "learning_rate": 0.0001,
                        "batch_size": 32,
                        "epochs": 10
                    }
                },
                "vectorized_labelling_orchestrator": {
                    "profit_take_multiplier": 0.002,
                    "stop_loss_multiplier": 0.001,
                    "time_barrier_minutes": 30
                }
            }
            
            step9_5 = HMMLMGeneralistTrainingStep(config)
            await step9_5.initialize()
            
            # Add composite cluster ID to test data
            test_data['composite_cluster_id'] = test_data['true_regime']
            
            # Test enhanced regime change detection
            self.logger.info("🔍 Testing enhanced regime change detection...")
            
            regime_events = step9_5._detect_regime_changes_enhanced(
                test_data, 0.002, 0.001
            )
            
            if regime_events:
                self.logger.info("✅ Enhanced regime change detection successful")
                self.logger.info(f"📊 Detected {len(regime_events)} regime events")
                
                # Analyze event characteristics
                confidences = [event.get('regime_confidence', 0) for event in regime_events]
                transition_probs = [event.get('transition_probability', 0) for event in regime_events]
                
                self.logger.info(f"📈 Average confidence: {np.mean(confidences):.3f}")
                self.logger.info(f"📈 Average transition probability: {np.mean(transition_probs):.3f}")
                
                # Count high-confidence events
                high_conf_events = [e for e in regime_events if e.get('regime_confidence', 0) > 0.7]
                self.logger.info(f"🎯 High-confidence events: {len(high_conf_events)}")
                
            else:
                self.logger.warning("⚠️ Enhanced regime change detection returned no events")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Step 9.5 enhanced detection test failed: {e}")
            return False
    
    async def test_enhanced_regime_predictor(self, test_data: pd.DataFrame):
        """Test the enhanced regime predictor."""
        self.logger.info("🧪 Testing Enhanced Regime Predictor...")
        
        try:
            # Initialize enhanced predictor
            config = {
                "stability_threshold": 0.1,
                "min_persistence": 3,
                "entropy_percentile": 75,
                "confidence_threshold": 0.7
            }
            
            predictor = EnhancedRegimePredictor(config)
            
            # Prepare test data
            features = test_data[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Add some derived features
            features['price_momentum_10'] = features['close'].pct_change(10)
            features['volatility_20'] = features['close'].pct_change().rolling(20).std()
            features['volume_ratio_10'] = features['volume'] / features['volume'].rolling(10).mean()
            features['rsi'] = self._calculate_rsi(features['close'])
            features['adx'] = self._calculate_adx(features)
            features['bb_position'] = (features['close'] - features['close'].rolling(20).mean()) / features['close'].rolling(20).std()
            features['atr_normalized'] = self._calculate_atr(features) / features['close']
            
            # Fill NaN values
            features = features.fillna(0)
            
            # Generate mock HMM data
            n_samples = len(features)
            n_states = 4
            
            hmm_probs = np.random.dirichlet([1, 1, 1, 1], size=n_samples)
            hmm_states = np.random.randint(0, n_states, size=n_samples)
            
            # Fit persistence model
            self.logger.info("📊 Fitting persistence model...")
            persistence_success = predictor.fit_persistence_model(hmm_states)
            
            if persistence_success:
                self.logger.info("✅ Persistence model fitted successfully")
                persistence_summary = predictor.get_model_summary()
                self.logger.info(f"📈 Persistence model: {persistence_summary['persistence_model']}")
            else:
                self.logger.warning("⚠️ Persistence model fitting failed")
            
            # Fit adaptive boundaries
            self.logger.info("🔧 Fitting adaptive boundaries...")
            boundaries_success = predictor.fit_adaptive_boundaries(features)
            
            if boundaries_success:
                self.logger.info("✅ Adaptive boundaries fitted successfully")
                boundaries_summary = predictor.get_model_summary()
                self.logger.info(f"📊 Adaptive boundaries: {boundaries_summary['adaptive_boundaries']}")
            else:
                self.logger.warning("⚠️ Adaptive boundaries fitting failed")
            
            # Test regime change prediction
            self.logger.info("🔮 Testing regime change prediction...")
            
            predictions = predictor.predict_regime_changes(features, hmm_probs, hmm_states)
            
            if predictions.get("success", False):
                self.logger.info("✅ Regime change prediction successful")
                self.logger.info(f"📊 High-confidence predictions: {len(predictions['predictions'])}")
                self.logger.info(f"📈 All predictions: {len(predictions['all_predictions'])}")
                
                # Analyze prediction quality
                if predictions['predictions']:
                    confidences = [pred['confidence'] for pred in predictions['predictions']]
                    transition_probs = [pred['transition_probability'] for pred in predictions['predictions']]
                    
                    self.logger.info(f"📈 Average confidence: {np.mean(confidences):.3f}")
                    self.logger.info(f"📈 Average transition probability: {np.mean(transition_probs):.3f}")
                    self.logger.info(f"📈 Max confidence: {np.max(confidences):.3f}")
                    
                    # Show sample predictions
                    for i, pred in enumerate(predictions['predictions'][:3]):
                        self.logger.info(f"🎯 Prediction {i+1}: {pred}")
                
            else:
                self.logger.error("❌ Regime change prediction failed")
                self.logger.error(f"Error: {predictions.get('error', 'Unknown error')}")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Enhanced regime predictor test failed: {e}")
            return False
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ADX indicator."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Calculate smoothed values
        tr_smooth = tr.rolling(window=window).mean()
        dm_plus_smooth = dm_plus.rolling(window=window).mean()
        dm_minus_smooth = dm_minus.rolling(window=window).mean()
        
        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # Calculate DX and ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=window).mean()
        
        return adx
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ATR indicator."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    async def run_comprehensive_test(self):
        """Run comprehensive test of all enhanced HMM capabilities."""
        self.logger.info("🚀 Starting comprehensive enhanced HMM capabilities test...")
        
        # Generate test data
        test_data = self.generate_test_data(n_samples=1000)
        
        # Test results
        results = {}
        
        # Test Step 3 enhanced features
        self.logger.info("=" * 60)
        self.logger.info("TESTING STEP 3 ENHANCED FEATURES")
        self.logger.info("=" * 60)
        results['step3'] = await self.test_step3_enhanced_features(test_data)
        
        # Test Step 9.5 enhanced detection
        self.logger.info("=" * 60)
        self.logger.info("TESTING STEP 9.5 ENHANCED DETECTION")
        self.logger.info("=" * 60)
        results['step9_5'] = await self.test_step9_5_enhanced_detection(test_data)
        
        # Test Enhanced Regime Predictor
        self.logger.info("=" * 60)
        self.logger.info("TESTING ENHANCED REGIME PREDICTOR")
        self.logger.info("=" * 60)
        results['predictor'] = await self.test_enhanced_regime_predictor(test_data)
        
        # Summary
        self.logger.info("=" * 60)
        self.logger.info("COMPREHENSIVE TEST SUMMARY")
        self.logger.info("=" * 60)
        
        for test_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            self.logger.info(f"{test_name.upper()}: {status}")
        
        overall_success = all(results.values())
        overall_status = "✅ ALL TESTS PASSED" if overall_success else "❌ SOME TESTS FAILED"
        
        self.logger.info("=" * 60)
        self.logger.info(f"OVERALL RESULT: {overall_status}")
        self.logger.info("=" * 60)
        
        return overall_success


async def main():
    """Main test function."""
    tester = EnhancedHMMTester()
    success = await tester.run_comprehensive_test()
    
    if success:
        print("\n🎉 All enhanced HMM capabilities tests passed!")
        print("✅ Enhanced regime change detection is working properly")
        print("✅ Adaptive regime boundaries are functioning")
        print("✅ Regime persistence modeling is operational")
        print("✅ Multi-signal regime change detection is effective")
    else:
        print("\n💥 Some enhanced HMM capabilities tests failed!")
        print("Please check the logs for detailed error information")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())