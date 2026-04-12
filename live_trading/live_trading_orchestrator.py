"""Live Trading Orchestrator.

Main orchestration logic that wires together:
1. Data feed from Binance (margin-enabled assets passing universe filters)
2. Feature generation (two-stage: masks first, then full ML features)
3. Model inference with confidence calibration
4. Portfolio management constraints
5. Order execution with OCO

Flow:
Get data from Binance -> Apply masks -> Compute features -> Run models ->
If confidence above threshold + portfolio allows -> Place order ->
Monitor until closing, managing other potential/current positions
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

import pandas as pd
import numpy as np

try:
    from extreme_price_movements.portfolio_manager import PortfolioManager
    from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
    from extreme_price_movements.inference.feature_generator import (
        get_features_for_candidates,
        _requires_gated_feature_generation,
    )
    from extreme_price_movements.offline_optimisers.params_store import (
        load_inference_candidate_mask_params_by_mode,
        load_inference_candidate_mask_params_per_bucket,
    )
    from extreme_price_movements.simple_position_sizer import (
        load_calibration_curves,
        calibrate_score,
    )
    from extreme_price_movements.inference.candidate_selector import _build_mask_for_mode, _up_down_zones
    from extreme_price_movements.utils import tprint
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from extreme_price_movements.portfolio_manager import PortfolioManager
    from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
    from extreme_price_movements.inference.feature_generator import (
        get_features_for_candidates,
        _requires_gated_feature_generation,
    )
    from extreme_price_movements.offline_optimisers.params_store import (
        load_inference_candidate_mask_params_by_mode,
        load_inference_candidate_mask_params_per_bucket,
    )
    from extreme_price_movements.simple_position_sizer import (
        load_calibration_curves,
        calibrate_score,
    )
    from extreme_price_movements.inference.candidate_selector import _build_mask_for_mode, _up_down_zones
    from extreme_price_movements.utils import tprint

# Local imports (same directory)
from binance_data_feed import BinanceDataFeed, DataFeedConfig
from order_manager_v2 import OrderManagerV2
try:
    from api_client import APIClient, TradingConfig
except ImportError:
    from live_trading.api_client import APIClient, TradingConfig


@dataclass
class LiveTradingConfig:
    """Configuration for live trading orchestrator."""
    # Data feed
    timeframe: str = "15m"
    lookback_bars: int = 200
    data_update_interval: float = 60.0
    
    # Strategy
    data_root: str = "."
    run_id: Optional[str] = None
    
    # Portfolio
    max_positions: int = 4
    max_portfolio_pct: float = 0.30
    max_position_usdt: float = 5000.0
    cooldown_hours: float = 24.0
    
    # Inference
    confidence_threshold: float = 0.5
    use_calibration: bool = True
    use_strategy_acceptance: bool = True
    min_calibrated_confidence: float = 0.75  # Top 75%
    
    # Execution
    initial_threshold: float = 0.5
    monitor_interval: float = 60.0
    
    # Safety
    emergency_stop_loss_pct: float = 0.05
    max_daily_trades: int = 20


class LiveTradingOrchestrator:
    """Main orchestrator for live trading.
    
    Coordinates the full trading pipeline:
    1. Fetches data from Binance for margin-enabled universe
    2. Applies masks and generates features (two-stage)
    3. Runs model inference with confidence calibration
    4. Applies PortfolioManager constraints
    5. Executes trades with OCO orders
    6. Monitors positions until close
    
    Compatible with PortfolioManager and calibration from previous implementation.
    """
    
    def __init__(
        self,
        config: LiveTradingConfig,
        model_bundle: Dict[str, Any],
        strategy_exit_params: Dict[str, Dict[str, Any]],
    ):
        self.config = config
        self.model_bundle = model_bundle
        self.strategy_exit_params = strategy_exit_params
        
        # Initialize components
        self.api_client: Optional[APIClient] = None
        self.data_feed: Optional[BinanceDataFeed] = None
        self.portfolio_mgr: Optional[PortfolioManager] = None
        self.order_manager: Optional[OrderManagerV2] = None
        self.model_orchestrator: Optional[ModelOrchestrator] = None
        
        # State
        self._running = False
        self._main_task: Optional[asyncio.Task] = None
        self.calibration_data: Dict[str, Dict[str, Any]] = {}
        self.accepted_strategies: Optional[Set[str]] = None
        self.mask_params_by_mode: Dict[str, Dict[str, Any]] = {}
        
        # Tracking
        self.daily_trade_count = 0
        self.last_trade_date: Optional[datetime] = None
        self.trade_history: List[Dict[str, Any]] = []
        
        # Callbacks
        self._on_signal: List[Callable[[Dict[str, Any]], None]] = []
        self._on_trade: List[Callable[[Dict[str, Any]], None]] = []
        self._on_error: List[Callable[[Exception], None]] = []
    
    async def initialize(self) -> None:
        """Initialize all components."""
        tprint("[Orchestrator] Initializing live trading orchestrator...")
        
        # 1. Initialize API Client
        trading_config = TradingConfig()
        trading_config.exchanges = {
            "binance": {
                "api_key": self._get_api_key(),
                "api_secret": self._get_api_secret(),
            }
        }
        self.api_client = APIClient(trading_config, "binance")
        await self.api_client.start()
        
        # 2. Initialize PortfolioManager
        self.portfolio_mgr = PortfolioManager(
            max_positions=self.config.max_positions,
            max_portfolio_pct=self.config.max_portfolio_pct,
            max_position_usdt=self.config.max_position_usdt,
            cooldown_hours=self.config.cooldown_hours,
            max_same_side_pct=0.75,
            max_same_strategy_pct=0.50,
        )
        tprint("[Orchestrator] PortfolioManager initialized")
        
        # 3. Initialize Data Feed
        data_config = DataFeedConfig(
            timeframe=self.config.timeframe,
            lookback_bars=self.config.lookback_bars,
            update_interval_seconds=self.config.data_update_interval,
            quotes=("USDT",),
        )
        self.data_feed = BinanceDataFeed(
            api_client=self.api_client,
            config=data_config,
        )
        await self.data_feed.initialize()
        tprint("[Orchestrator] DataFeed initialized")
        
        # 4. Initialize Order Manager
        order_config = {
            "sl_mult": 1.0,
            "tp_mult": 3.0,
            "trail_mult": 0.25,
            "monitor_interval_seconds": self.config.monitor_interval,
            "fee_rate": 0.0003,
        }
        self.order_manager = OrderManagerV2(
            api_client=self.api_client,
            portfolio_manager=self.portfolio_mgr,
            config=order_config,
        )
        await self.order_manager.start()
        tprint("[Orchestrator] OrderManager initialized")
        
        # 5. Initialize Model Orchestrator
        self.model_orchestrator = ModelOrchestrator(
            model_bundle=self.model_bundle,
            runtime_cfg={
                "entry_policy_config": self.strategy_exit_params.get("entry_policy"),
            }
        )
        tprint("[Orchestrator] ModelOrchestrator initialized")
        
        # 6. Load calibration and acceptance data
        await self._load_strategy_data()
        
        # 7. Load mask parameters
        self.mask_params_by_mode = load_inference_candidate_mask_params_by_mode()
        tprint(f"[Orchestrator] Loaded mask params for {len(self.mask_params_by_mode)} modes")
        
        # 8. Register data callback
        self.data_feed.register_data_callback(self._on_data_update)
        
        tprint("[Orchestrator] Initialization complete")
    
    def _get_api_key(self) -> str:
        """Get Binance API key from environment or config."""
        import os
        return os.environ.get("BINANCE_API_KEY", "")
    
    def _get_api_secret(self) -> str:
        """Get Binance API secret from environment or config."""
        import os
        return os.environ.get("BINANCE_API_SECRET", "")
    
    async def _load_strategy_data(self) -> None:
        """Load calibration and strategy acceptance data."""
        # Load calibration curves
        if self.config.use_calibration and self.config.run_id:
            self.calibration_data = load_calibration_curves(
                self.config.data_root,
                self.config.run_id
            )
            tprint(f"[Orchestrator] Loaded calibration for {len(self.calibration_data)} strategies")
        
        # Load strategy acceptance filter
        if self.config.use_strategy_acceptance and self.config.run_id:
            acceptance_path = Path(self.config.data_root) / "artifacts" / self.config.run_id / "strategy_final_acceptation.json"
            if acceptance_path.exists():
                try:
                    payload = json.loads(acceptance_path.read_text())
                    strategies = payload.get("strategies", [])
                    self.accepted_strategies = {s["strategy_id"] for s in strategies if "strategy_id" in s}
                    tprint(f"[Orchestrator] Loaded {len(self.accepted_strategies)} accepted strategies")
                except Exception as e:
                    tprint(f"[Orchestrator] Error loading strategy acceptance: {e}")
    
    async def start(self) -> None:
        """Start the live trading orchestrator."""
        if self._running:
            return
        
        self._running = True
        
        # Start data feed
        await self.data_feed.start()
        
        # Start main trading loop
        self._main_task = asyncio.create_task(self._trading_loop())
        
        tprint("[Orchestrator] Live trading started")
    
    async def stop(self) -> None:
        """Stop the live trading orchestrator."""
        self._running = False
        
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        
        if self.data_feed:
            await self.data_feed.stop()
        
        if self.order_manager:
            await self.order_manager.stop()
        
        if self.api_client:
            await self.api_client.stop()
        
        tprint("[Orchestrator] Live trading stopped")
    
    async def _trading_loop(self) -> None:
        """Main trading loop."""
        while self._running:
            try:
                # Reset daily trade count if new day
                await self._check_daily_reset()
                
                # Run inference cycle
                await self._run_inference_cycle()
                
                # Wait for next cycle
                await asyncio.sleep(self.config.data_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                tprint(f"[Orchestrator] Trading loop error: {e}")
                for callback in self._on_error:
                    callback(e)
                await asyncio.sleep(self.config.data_update_interval)
    
    async def _check_daily_reset(self) -> None:
        """Check if we need to reset daily counters."""
        now = datetime.utcnow()
        if self.last_trade_date is None or now.date() != self.last_trade_date:
            self.daily_trade_count = 0
            self.last_trade_date = now.date()
            tprint(f"[Orchestrator] New trading day: {self.last_trade_date}")
    
    async def _run_inference_cycle(self) -> None:
        """Run one inference cycle over all symbols."""
        # Get current data panel
        panel = self.data_feed.get_panel()
        if not panel:
            return
        
        # Get symbols to analyze
        symbols = list(panel.keys())
        
        # Stage 1: Generate mask features for all symbols
        mask_features = await self._generate_mask_features(panel, symbols)
        
        # Apply masks to get passing symbols
        passing_symbols = self._apply_masks(panel, mask_features, symbols)
        
        if not passing_symbols:
            return
        
        # Stage 2: Generate full ML features for passing symbols only
        full_features = await self._generate_full_features(panel, passing_symbols)
        
        # Run inference on each passing symbol
        candidates = []
        for symbol in passing_symbols:
            if symbol not in full_features:
                continue
            
            features = full_features[symbol]
            
            # Run model inference
            inference_result = self._run_inference(symbol, features)
            
            if inference_result:
                candidates.append(inference_result)
        
        if not candidates:
            return
        
        # Apply calibration and filter
        filtered_candidates = self._filter_by_calibration(candidates)
        
        if not filtered_candidates:
            return
        
        # Sort by confidence
        filtered_candidates.sort(key=lambda x: x.get("calibrated_confidence", 0), reverse=True)
        
        # Try to execute top candidates
        for candidate in filtered_candidates:
            if self.daily_trade_count >= self.config.max_daily_trades:
                break
            
            result = await self._execute_candidate(candidate)
            
            if result.get("success"):
                self.daily_trade_count += 1
                self.trade_history.append(result)
                
                # Only execute one trade per cycle to avoid over-trading
                break
    
    async def _generate_mask_features(
        self,
        panel: Dict[str, pd.DataFrame],
        symbols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Stage 1: Generate mask features (lightweight)."""
        mask_features = {}
        
        for symbol in symbols:
            if symbol not in panel:
                continue
            
            try:
                df = panel[symbol]
                
                # Compute basic features needed for masks
                # This is a simplified version - in production, use proper feature generator
                features = pd.DataFrame(index=df.index)
                
                # Price-based features
                features["close"] = df["close"]
                features["returns_1h"] = df["close"].pct_change(4)  # 4 * 15m = 1h
                features["returns_4h"] = df["close"].pct_change(16)
                features["returns_24h"] = df["close"].pct_change(96)
                
                # Volatility features
                features["volatility_20"] = df["close"].pct_change().rolling(20).std()
                features["atr_14"] = self._compute_atr(df, 14)
                
                # Volume features
                features["volume_ma_20"] = df["volume"].rolling(20).mean()
                features["volume_ratio"] = df["volume"] / features["volume_ma_20"]
                
                mask_features[symbol] = features
                
            except Exception as e:
                tprint(f"[Orchestrator] Error generating mask features for {symbol}: {e}")
        
        return mask_features
    
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def _apply_masks(
        self,
        panel: Dict[str, pd.DataFrame],
        mask_features: Dict[str, pd.DataFrame],
        symbols: List[str]
    ) -> List[str]:
        """Apply mask rules to filter symbols."""
        passing_symbols = []
        
        for symbol in symbols:
            if symbol not in mask_features:
                continue
            
            features = mask_features[symbol]
            
            # Get latest values
            if features.empty:
                continue
            
            latest = features.iloc[-1]
            
            # Apply simple mask logic (can be expanded)
            # For now, basic liquidity and volatility filters
            passes = True
            
            # Minimum volatility filter
            if latest.get("volatility_20", 0) < 0.001:  # Less than 0.1% volatility
                passes = False
            
            # Volume filter
            if latest.get("volume_ratio", 0) < 0.5:  # Below average volume
                passes = False
            
            if passes:
                passing_symbols.append(symbol)
        
        return passing_symbols
    
    async def _generate_full_features(
        self,
        panel: Dict[str, pd.DataFrame],
        symbols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Stage 2: Generate full ML features for passing symbols only."""
        full_features = {}
        
        # Use existing feature generator if available
        try:
            # Try to use the inference feature generator
            from extreme_price_movements.inference.feature_generator import get_features_for_candidates
            
            # Convert panel to format expected by feature generator
            features_dict = get_features_for_candidates(panel, symbols)
            
            if isinstance(features_dict, dict):
                for symbol in symbols:
                    if symbol in features_dict:
                        full_features[symbol] = features_dict[symbol]
            
        except Exception as e:
            tprint(f"[Orchestrator] Feature generator error, using fallback: {e}")
            
            # Fallback: use mask features as full features
            for symbol in symbols:
                if symbol in panel:
                    # Generate extended features
                    df = panel[symbol]
                    features = self._generate_extended_features(df)
                    full_features[symbol] = features
        
        return full_features
    
    def _generate_extended_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate extended feature set for inference."""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features["close"] = df["close"]
        features["open"] = df["open"]
        features["high"] = df["high"]
        features["low"] = df["low"]
        
        # Returns at different horizons
        for horizon in [1, 4, 16, 96]:
            features[f"returns_{horizon}"] = df["close"].pct_change(horizon)
        
        # Moving averages
        for window in [20, 50, 200]:
            features[f"sma_{window}"] = df["close"].rolling(window).mean()
            features[f"ema_{window}"] = df["close"].ewm(span=window).mean()
        
        # Volatility
        for window in [10, 20, 50]:
            features[f"volatility_{window}"] = df["close"].pct_change().rolling(window).std()
        
        # Volume features
        features["volume"] = df["volume"]
        features["volume_ma_20"] = df["volume"].rolling(20).mean()
        features["volume_ratio"] = df["volume"] / features["volume_ma_20"]
        
        # ATR
        features["atr_14"] = self._compute_atr(df, 14)
        features["atr_50"] = self._compute_atr(df, 50)
        
        # RSI
        features["rsi_14"] = self._compute_rsi(df["close"], 14)
        
        return features
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _run_inference(self, symbol: str, features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Run model inference on a symbol."""
        try:
            if features.empty:
                return None
            
            # Get latest features as single row
            latest_features = features.iloc[-1:]
            
            # Use ModelOrchestrator to run full inference chain
            # This includes: spike filter -> alpha models -> meta model -> position sizing
            results = self.model_orchestrator.run_inference_for_symbol(
                symbol=symbol,
                features=latest_features
            )
            
            if not results:
                return None
            
            # Extract key predictions
            confidence = results.get("meta_confidence", 0.0)
            side = results.get("predicted_side", "long")
            strategy_id = results.get("strategy_id", "default")
            
            # Check strategy acceptance
            if self.accepted_strategies and strategy_id not in self.accepted_strategies:
                return None
            
            # Get entry price
            entry_price = float(latest_features["close"].iloc[-1])
            
            return {
                "symbol": symbol,
                "side": side,
                "strategy_id": strategy_id,
                "confidence": confidence,
                "entry_price": entry_price,
                "features": latest_features,
                "raw_results": results,
            }
            
        except Exception as e:
            tprint(f"[Orchestrator] Inference error for {symbol}: {e}")
            return None
    
    def _filter_by_calibration(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter candidates by calibrated confidence."""
        if not self.calibration_data:
            # No calibration data, use raw confidence
            for candidate in candidates:
                candidate["calibrated_confidence"] = candidate.get("confidence", 0.0)
            return [c for c in candidates if c["confidence"] >= self.config.confidence_threshold]
        
        filtered = []
        
        for candidate in candidates:
            strategy_id = candidate.get("strategy_id", "default")
            raw_confidence = candidate.get("confidence", 0.0)
            
            # Calibrate confidence
            calibrated = calibrate_score(raw_confidence, strategy_id, self.calibration_data)
            candidate["calibrated_confidence"] = calibrated
            
            # Get threshold for this strategy
            calib = self.calibration_data.get(strategy_id, {})
            threshold = calib.get("p75_threshold", self.config.confidence_threshold)
            
            # Check if passes threshold
            if calibrated >= threshold:
                filtered.append(candidate)
        
        return filtered
    
    async def _execute_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade candidate."""
        symbol = candidate["symbol"]
        side = candidate["side"]
        strategy_id = candidate["strategy_id"]
        confidence = candidate.get("calibrated_confidence", candidate.get("confidence", 0.0))
        entry_price = candidate["entry_price"]
        
        # Get strategy parameters
        params = self.strategy_exit_params.get(strategy_id, {})
        
        # Check if we already have a position
        existing_position = self.order_manager.get_position(symbol)
        if existing_position:
            return {"success": False, "error": "Position already exists"}
        
        # Place OCO order
        result = await self.order_manager.place_oco_order(
            symbol=symbol,
            side=side,
            strategy_id=strategy_id,
            entry_price=entry_price,
            confidence_score=confidence,
            initial_threshold=self.config.initial_threshold,
            params=params,
        )
        
        if result.get("success"):
            tprint(f"[Orchestrator] Executed trade: {symbol} {side} at {entry_price:.4f}")
            
            # Notify callbacks
            for callback in self._on_trade:
                callback(result)
        else:
            tprint(f"[Orchestrator] Trade failed: {result.get('error', 'unknown')}")
        
        return result
    
    def _on_data_update(self, symbol: str, data: pd.DataFrame) -> None:
        """Callback for data feed updates."""
        # Data feed has updated, next cycle will pick up new data
        pass
    
    async def emergency_stop(self) -> None:
        """Emergency stop - close all positions."""
        tprint("[Orchestrator] EMERGENCY STOP triggered")
        
        if self.order_manager:
            await self.order_manager.emergency_close_all()
        
        await self.stop()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        return {
            "running": self._running,
            "daily_trade_count": self.daily_trade_count,
            "max_daily_trades": self.config.max_daily_trades,
            "portfolio_state": self.portfolio_mgr.get_portfolio_state() if self.portfolio_mgr else None,
            "open_positions": len(self.order_manager.get_all_positions()) if self.order_manager else 0,
            "data_feed_symbols": len(self.data_feed.trading_symbols) if self.data_feed else 0,
            "calibration_loaded": len(self.calibration_data),
            "strategies_accepted": len(self.accepted_strategies) if self.accepted_strategies else 0,
        }
    
    def register_signal_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for trading signals."""
        self._on_signal.append(callback)
    
    def register_trade_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for executed trades."""
        self._on_trade.append(callback)


async def main():
    """Main entry point for live trading."""
    # Load configuration
    config = LiveTradingConfig(
        data_root=".",
        run_id=None,  # Set to latest run
        confidence_threshold=0.5,
    )
    
    # Load model bundle
    # This would come from model loader
    model_bundle = {}
    
    # Load strategy parameters
    strategy_exit_params = {}
    
    # Create orchestrator
    orchestrator = LiveTradingOrchestrator(
        config=config,
        model_bundle=model_bundle,
        strategy_exit_params=strategy_exit_params,
    )
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        asyncio.create_task(orchestrator.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize and start
        await orchestrator.initialize()
        await orchestrator.start()
        
        # Run forever
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        tprint(f"[Main] Fatal error: {e}")
        await orchestrator.emergency_stop()
        raise


if __name__ == "__main__":
    asyncio.run(main())


__all__ = ["LiveTradingOrchestrator", "LiveTradingConfig"]
