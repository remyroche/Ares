import asyncio
import pandas as pd
from loguru import logger
from collections import deque

from .technical_analyzer import TechnicalAnalyzer
from .regime_classifier import RegimeClassifier

class Analyst:
    """
    The central orchestrator for the analysis pipeline.

    It receives raw market data, coordinates specialized sub-components
    (like TechnicalAnalyzer and RegimeClassifier) to process it, and
    forwards a comprehensive analysis object to the Strategist.
    """

    def __init__(self, kline_history_size=500):
        self.logger = logger
        self.technical_analyzer = TechnicalAnalyzer()
        self.regime_classifier = RegimeClassifier()
        
        # Use a deque for efficient fixed-size storage of kline history
        self.kline_history = deque(maxlen=kline_history_size)
        self.logger.info("Analyst orchestrator initialized.")

    async def run(self, market_data_queue: asyncio.Queue, analysis_queue: asyncio.Queue):
        """
        The main async loop for the Analyst.
        It continuously processes data from the market queue and pushes
        analysis to the analysis queue.
        """
        self.logger.info("Analyst task started. Waiting for market data...")
        while True:
            try:
                # Wait for new market data from the Sentinel
                market_data = await market_data_queue.get()
                
                if market_data['type'] == 'kline':
                    await self._process_kline_data(market_data['data'], analysis_queue)
                
                # Acknowledge that the task is done
                market_data_queue.task_done()

            except asyncio.CancelledError:
                self.logger.info("Analyst task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"An error occurred in the Analyst run loop: {e}", exc_info=True)
                # Avoid crashing the loop on a single bad data point
                await asyncio.sleep(1)

    async def _process_kline_data(self, kline_data: dict, analysis_queue: asyncio.Queue):
        """Processes a single kline update."""
        
        # Append new kline to our history.
        # This assumes kline_data is a dictionary for a single candle.
        self.kline_history.append(kline_data)

        # Convert the history to a DataFrame for analysis
        # This is inefficient to do on every tick; a more optimized version might
        # update the last row or use a streaming-compatible library. For clarity,
        # we recreate it here.
        df = pd.DataFrame(list(self.kline_history))
        # Ensure correct data types for analysis
        df = df.astype({
            'open': 'float', 'high': 'float', 'low': 'float', 
            'close': 'float', 'volume': 'float'
        })


        # --- Analysis Pipeline ---
        # 1. Calculate technical indicators
        df_with_ta = self.technical_analyzer.calculate_indicators(df)

        # 2. Classify the market regime
        regime = self.regime_classifier.classify(df_with_ta)

        # 3. Assemble the final analysis object
        # We only care about the analysis of the *most recent* data point
        latest_analysis = df_with_ta.iloc[-1].to_dict()
        
        analysis_package = {
            'timestamp': pd.to_datetime(kline_data['T'], unit='ms'),
            'type': 'full_analysis',
            'market_regime': regime,
            'indicators': latest_analysis,
            'raw_kline': kline_data
        }
        
        # 4. Push the analysis to the Strategist
        await analysis_queue.put(analysis_package)
        self.logger.debug(f"Analysis package sent. Regime: {regime}")
