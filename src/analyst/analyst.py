import asyncio
import pandas as pd
from typing import Dict, Any

from src.exchange.binance import BinanceExchange
from src.utils.logger import logger
from src.config import settings
from src.utils.state_manager import StateManager
from src.analyst.feature_engineering import FeatureEngineering
from src.analyst.regime_classifier import RegimeClassifier
from src.analyst.sr_analyzer import SRAnalyzer
from src.analyst.technical_analyzer import TechnicalAnalyzer
from src.analyst.market_health_analyzer import MarketHealthAnalyzer
from src.analyst.liquidation_risk_model import LiquidationRiskModel

class Analyst:
    """
    The Analyst processes real-time and historical market data to generate actionable intelligence.
    It operates in an event-driven manner, triggering its analysis pipeline upon the
    closure of a new candlestick.
    """

    def __init__(self, exchange_client: BinanceExchange, state_manager: StateManager):
        """
        Initializes the Analyst.

        Args:
            exchange_client: An instance of the BinanceExchange client to access data.
            state_manager: An instance of the StateManager to save analysis results.
        """
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.logger = logger.getChild('Analyst')
        self.trade_symbol = settings.trade_symbol
        self.timeframe = settings.timeframe
        self.last_kline_open_time = None

        # Instantiate all sub-analyzer components
        self.feature_engineering = FeatureEngineering()
        self.regime_classifier = RegimeClassifier()
        self.sr_analyzer = SRAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.market_health_analyzer = MarketHealthAnalyzer()
        self.liquidation_risk_model = LiquidationRiskModel()
        self.logger.info("Analyst and all sub-analyzers initialized.")

    async def start(self):
        """
        Starts the main analysis loop.
        The loop waits for a new kline to close before running the full analysis pipeline.
        """
        self.logger.info("Analyst started. Waiting for new kline events...")
        while True:
            try:
                # Wait for the next kline to be available from the WebSocket stream
                latest_kline = self.exchange.kline_data
                if latest_kline and latest_kline.get('is_closed'):
                    # Check if this is a new kline we haven't processed yet
                    if latest_kline['open_time'] != self.last_kline_open_time:
                        self.last_kline_open_time = latest_kline['open_time']
                        self.logger.info(f"New kline closed at {pd.to_datetime(self.last_kline_open_time, unit='ms')}. Triggering analysis.")
                        
                        # Run the analysis pipeline
                        await self.run_analysis_pipeline()
                
                # Sleep for a short duration to prevent a tight loop
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                self.logger.info("Analyst task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"An error occurred in the Analyst loop: {e}", exc_info=True)
                # Wait before retrying to prevent rapid failure loops
                await asyncio.sleep(10)

    async def run_analysis_pipeline(self):
        """
        Executes the full sequence of analysis tasks.
        """
        self.logger.info("--- Starting Analysis Pipeline ---")
        try:
            # 1. Fetch necessary historical data
            klines = await self.exchange.get_klines(self.trade_symbol, self.timeframe, limit=500)
            if not klines:
                self.logger.error("Could not fetch historical klines. Aborting analysis cycle.")
                return

            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            # 2. Get real-time data from the exchange client's state
            order_book = self.exchange.order_book
            recent_trades = self.exchange.recent_trades

            # 3. Run all analysis components in sequence
            df_features = self.feature_engineering.add_features(df.copy())
            market_regime = self.regime_classifier.classify(df_features)
            sr_levels = self.sr_analyzer.analyze(df_features)
            technical_signals = self.technical_analyzer.analyze(df_features)
            market_health = self.market_health_analyzer.analyze(order_book, recent_trades)
            liquidation_risk = self.liquidation_risk_model.calculate(df_features, sr_levels)

            # 4. Consolidate intelligence
            analyst_intelligence = {
                "timestamp": int(time.time() * 1000),
                "market_regime": market_regime,
                "support_resistance": sr_levels,
                "technical_signals": technical_signals,
                "market_health": market_health,
                "liquidation_risk_score": liquidation_risk,
                # Simple directional confidence for demonstration
                "directional_confidence_score": self._calculate_confidence(technical_signals, market_regime)
            }

            # 5. Save the final intelligence packet to the StateManager
            self.state_manager.set_state("analyst_intelligence", analyst_intelligence)
            self.logger.info(f"Analysis complete. Intelligence package updated. Regime: {market_regime}")
            self.logger.debug(f"Analyst Intelligence: {analyst_intelligence}")

        except Exception as e:
            self.logger.error(f"Error during analysis pipeline: {e}", exc_info=True)

    def _calculate_confidence(self, signals: Dict, regime: str) -> float:
        """
        A simple heuristic to generate a directional confidence score.
        This should be replaced with a more sophisticated model.
        """
        score = 0.5  # Neutral baseline
        if "BULL" in regime:
            score += 0.2
        elif "BEAR" in regime:
            score -= 0.2
        
        if signals.get('rsi_signal') == 'buy':
            score += 0.1
        elif signals.get('rsi_signal') == 'sell':
            score -= 0.1

        if signals.get('macd_signal') == 'buy':
            score += 0.1
        elif signals.get('macd_signal') == 'sell':
            score -= 0.1
            
        return max(0, min(1, round(score, 2))) # Clamp between 0 and 1
