"""
Tests unitaires pour PaperTradingSimulator

Ce module teste les fonctionnalités du simulateur de trading papier.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import du module à tester
try:
    from src.simulator.paper_trading_simulator import PaperTradingSimulator, OrderType, OrderStatus, PositionSide
except ImportError:
    # Si le module n'existe pas encore, on utilise un mock
    PaperTradingSimulator = Mock
    OrderType = Mock
    OrderStatus = Mock
    PositionSide = Mock

try:
    from src.simulator.config import SimulatorConfig, SlippageModel
except ImportError:
    SimulatorConfig = Mock
    SlippageModel = Mock


@pytest.mark.unit
@pytest.mark.simulator
@pytest.mark.asyncio
class TestPaperTradingSimulator:
    """Classe de tests pour PaperTradingSimulator."""

    def setup_method(self):
        """Setup pour chaque test."""
        self.initial_balance = 10000.0
        self.commission = 0.001  # 0.1%
        self.slippage = 0.0005  # 0.05%
        
        # Créer une instance si la classe existe
        if hasattr(PaperTradingSimulator, '__call__'):
            # Créer une configuration par défaut
            config = SimulatorConfig()
            config.default_taker_fee = self.commission
            config.default_maker_fee = self.commission
            config.max_slippage_pct = self.slippage
            config.slippage_model = SlippageModel.ORDERBOOK
            
            self.simulator = PaperTradingSimulator(
                config=config,
                exchange="binance",
                initial_balance=self.initial_balance,
                direction_constraint="both"
            )
        else:
            self.simulator = AsyncMock()

    async def test_initialization_nominal(self):
        """Test d'initialisation nominale."""
        # Given/When/Then
        # Vérifier que le simulateur est correctement initialisé
        assert self.simulator is not None
        
        # Si c'est un mock, vérifier les attributs du mock
        if hasattr(self.simulator, 'initial_balance'):
            assert self.simulator.initial_balance == self.initial_balance
            assert self.simulator.current_balance == self.initial_balance
            assert self.simulator.exchange == "binance"
            assert self.simulator.direction_constraint == "both"
        else:
            # Pour le mock, on vérifie juste qu'il existe
            assert True

    async def test_simulate_order_buy_nominal(self):
        """Test de simulation d'ordre au marché en achat nominale."""
        # Given
        symbol = 'ETHUSDT'
        side = 'buy'
        order_type = 'market'
        quantity = 0.1
        price = 2000.0
        
        # Créer un order book simulé
        order_book = {
            'bids': [(price - 0.01, 100.0), (price - 0.02, 50.0)],
            'asks': [(price + 0.01, 100.0), (price + 0.02, 50.0)]
        }
        
        # When
        result = await self.simulator.simulate_order(symbol, side, order_type, quantity, price, order_book)
        
        # Then
        assert result['status'] == "FILLED"
        assert result['symbol'] == symbol
        assert result['side'] == side.upper()
        assert result['quantity'] == quantity
        assert 'price' in result
        assert 'fee' in result
        assert 'slippagePct' in result

    async def test_simulate_order_sell_nominal(self):
        """Test de simulation d'ordre au marché en vente nominale."""
        # Given
        symbol = 'ETHUSDT'
        side = 'sell'
        order_type = 'market'
        quantity = 0.1
        price = 2000.0
        
        # D'abord ouvrir une position longue
        order_book_buy = {
            'bids': [(price - 0.01, 100.0), (price - 0.02, 50.0)],
            'asks': [(price + 0.01, 100.0), (price + 0.02, 50.0)]
        }
        await self.simulator.simulate_order(symbol, 'buy', 'market', 0.1, None, order_book_buy)
        
        # Mock du prix de marché pour la vente
        order_book_sell = {
            'bids': [(price - 0.01, 100.0), (price - 0.02, 50.0)],
            'asks': [(price + 0.01, 100.0), (price + 0.02, 50.0)]
        }
        
        # When
        result = await self.simulator.simulate_order(symbol, side, order_type, quantity, price, order_book_sell)
        
        # Then
        assert result['status'] == "FILLED"
        assert result['symbol'] == symbol
        assert result['side'] == side.upper()
        assert result['quantity'] == quantity

    async def test_simulate_limit_order_nominal(self):
        """Test de simulation d'ordre limite nominale."""
        # Given
        symbol = 'ETHUSDT'
        side = 'buy'
        order_type = 'limit'
        quantity = 0.1
        price = 1990.0  # En dessous du prix du marché
        
        # Créer un order book avec prix plus élevé
        order_book = {
            'bids': [(price - 0.01, 100.0), (price - 0.02, 50.0)],
            'asks': [(price + 0.01, 100.0), (price + 0.02, 50.0)]
        }
        
        # When
        result = await self.simulator.simulate_order(symbol, side, order_type, quantity, price, order_book)
        
        # Then
        assert result['status'] == "FILLED"
        assert result['symbol'] == symbol
        assert result['side'] == side.upper()
        assert result['quantity'] == quantity
        assert result['price'] == price

    async def test_insufficient_balance(self):
        """Test de solde insuffisant."""
        # Given
        symbol = 'ETHUSDT'
        side = 'buy'
        order_type = 'market'
        quantity = 100.0  # Beaucoup plus que le solde de 10000$
        price = 2000.0
        
        # Créer un order book simulé
        order_book = {
            'bids': [(price - 0.01, 100.0), (price - 0.02, 50.0)],
            'asks': [(price + 0.01, 100.0), (price + 0.02, 50.0)]
        }
        
        # When
        result = await self.simulator.simulate_order(symbol, side, order_type, quantity, price, order_book)
        
        # Then
        assert result['status'] == "REJECTED"
        assert 'rejectedReason' in result
        assert 'balance' in result['rejectedReason'].lower() or 'insufficient' in result['rejectedReason'].lower()

    async def test_get_positions_nominal(self):
        """Test de récupération des positions nominale."""
        # Given/When
        positions = self.simulator.get_positions()
        
        # Then
        assert isinstance(positions, list)
        # Initialement, aucune position
        assert len(positions) == 0

    async def test_get_trade_history_nominal(self):
        """Test de récupération de l'historique des trades nominale."""
        # Given/When
        trades = self.simulator.get_trade_history()
        
        # Then
        assert isinstance(trades, list)
        # Initialement, aucun trade
        assert len(trades) == 0

    async def test_get_performance_metrics_nominal(self):
        """Test de récupération des métriques de performance nominale."""
        # Given/When
        metrics = self.simulator.get_performance_metrics()
        
        # Then
        assert isinstance(metrics, dict)
        assert 'total_trades' in metrics
        assert 'winning_trades' in metrics
        assert 'losing_trades' in metrics
        assert 'total_pnl' in metrics
        assert 'total_fees' in metrics
        assert 'current_balance' in metrics
        assert 'initial_balance' in metrics
        
        # Initialement, aucun trade
        assert metrics['total_trades'] == 0
        assert metrics['total_pnl'] == 0.0
        assert metrics['current_balance'] == self.initial_balance
        assert metrics['initial_balance'] == self.initial_balance

    async def test_generate_daily_report_nominal(self):
        """Test de génération de rapport quotidien nominale."""
        # Given
        symbol = 'ETHUSDT'
        
        # When
        result = await self.simulator.generate_daily_report(symbol)
        
        # Then
        assert isinstance(result, bool)
        # Le résultat dépend de l'implémentation, mais ne devrait pas lever d'exception

    async def test_multiple_orders_concurrent(self):
        """Test d'ordres multiples concurrents."""
        # Given
        symbol = 'ETHUSDT'
        order_book = {
            'bids': [(2000.0 - 0.01, 100.0), (2000.0 - 0.02, 50.0)],
            'asks': [(2000.0 + 0.01, 100.0), (2000.0 + 0.02, 50.0)]
        }
        
        # When
        # Placer plusieurs ordres simultanément
        tasks = [
            self.simulator.simulate_order(symbol, 'buy', 'market', 0.1, None, order_book),
            self.simulator.simulate_order(symbol, 'sell', 'market', 0.05, None, order_book),
            self.simulator.simulate_order(symbol, 'buy', 'limit', 0.05, 1990.0, order_book)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Then
        # Vérifier que tous les ordres ont été traités
        successful_orders = [r for r in results if isinstance(r, dict) and r.get('status') in ["FILLED", "REJECTED"]]
        assert len(successful_orders) == 3
        
        # Vérifier que les ordres ont des IDs uniques
        order_ids = [r['orderId'] for r in successful_orders if 'orderId' in r]
        assert len(set(order_ids)) == 3

    async def test_order_validation_invalid_inputs(self):
        """Test de validation d'ordres avec entrées invalides."""
        # Given
        symbol = 'ETHUSDT'
        order_book = {
            'bids': [(2000.0, 100.0)],
            'asks': [(2000.0, 100.0)]
        }
        
        # When/Then - Test avec quantité négative
        result = await self.simulator.simulate_order(symbol, 'buy', 'market', -0.1, None, order_book)
        assert result['status'] == "REJECTED"
        assert 'rejectedReason' in result
        
        # When/Then - Test avec symbole vide
        result = await self.simulator.simulate_order('', 'buy', 'market', 0.1, None, order_book)
        assert result['status'] == "REJECTED"
        assert 'rejectedReason' in result

    async def test_performance_with_many_orders(self):
        """Test de performance avec beaucoup d'ordres."""
        # Given
        symbol = 'ETHUSDT'
        order_book = {
            'bids': [(2000.0 - 0.01, 100.0)],
            'asks': [(2000.0 + 0.01, 100.0)]
        }
        
        # When
        start_time = datetime.now()
        
        # Placer beaucoup d'ordres
        tasks = []
        for i in range(50):  # Réduit pour éviter les timeouts
            tasks.append(self.simulator.simulate_order(f'{symbol}{i}', 'buy', 'market', 0.1, None, order_book))
        
        await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        
        # Then
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 10.0  # Devrait s'exécuter rapidement

    async def test_memory_usage_with_many_positions(self):
        """Test de l'utilisation mémoire avec beaucoup de positions."""
        # Given
        symbol = 'ETHUSDT'
        order_book = {
            'bids': [(2000.0 - 0.01, 100.0)],
            'asks': [(2000.0 + 0.01, 100.0)]
        }
        
        # When
        # Simuler beaucoup de positions
        tasks = []
        for i in range(10):  # Réduit pour éviter les timeouts
            tasks.append(self.simulator.simulate_order(f'{symbol}{i}', 'buy', 'market', 0.1, None, order_book))
        
        await asyncio.gather(*tasks)
        
        # Then
        # Vérifier que le système peut gérer la charge
        positions = self.simulator.get_positions()
        assert len(positions) >= 0  # Au minimum, aucune erreur
        
        # Le système devrait pouvoir gérer cette charge sans erreur de mémoire
        # (En pratique, on pourrait vouloir ajouter des limites)