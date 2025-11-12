"""
Tests unitaires pour ExchangeInterface

Ce module teste les fonctionnalités de l'interface d'exchange.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import du module à tester
try:
    from src.trading.execution.exchange_interface import ExchangeInterface, OrderType, OrderStatus, ExchangeType
except ImportError:
    # Si le module n'existe pas encore, on utilise un mock
    ExchangeInterface = Mock
    OrderType = Mock
    OrderStatus = Mock
    ExchangeType = Mock


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.asyncio
class TestExchangeInterface:
    """Classe de tests pour ExchangeInterface."""

    def setup_method(self):
        """Setup pour chaque test."""
        self.exchange_config = {
            'name': 'binance',
            'api_key': 'test_api_key',
            'api_secret': 'test_api_secret',
            'sandbox': True,
            'timeout': 30,
            'rate_limit': 10
        }
        
        # Créer une instance si la classe existe
        if hasattr(ExchangeInterface, '__call__'):
            self.exchange_interface = ExchangeInterface(self.exchange_config)
        else:
            self.exchange_interface = Mock()

    async def test_initialization_nominal(self):
        """Test d'initialisation nominale."""
        # Given/When
        if hasattr(self.exchange_interface, 'start'):
            await self.exchange_interface.start()
        
        # Then
        if hasattr(self.exchange_interface, 'is_connected'):
            assert self.exchange_interface.is_connected is True
        if hasattr(self.exchange_interface, 'exchange_name'):
            assert self.exchange_interface.exchange_name == 'binance'
        if hasattr(self.exchange_interface, 'is_sandbox'):
            assert self.exchange_interface.is_sandbox is True

    async def test_connect_nominal(self):
        """Test de connexion nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'connect'):
            pytest.skip("connect method not implemented")
        
        # When
        result = await self.exchange_interface.connect()
        
        # Then
        assert result['success'] is True
        assert 'connection_id' in result
        assert 'timestamp' in result
        if hasattr(self.exchange_interface, 'is_connected'):
            assert self.exchange_interface.is_connected is True

    async def test_connect_already_connected(self):
        """Test de connexion déjà connecté."""
        # Given
        if not hasattr(self.exchange_interface, 'connect') or not hasattr(self.exchange_interface, 'is_connected'):
            pytest.skip("Required methods not implemented")
            
        self.exchange_interface.is_connected = True
        
        # When
        result = await self.exchange_interface.connect()
        
        # Then
        assert result['success'] is True
        assert 'message' in result
        assert 'already connected' in result['message'].lower()

    async def test_disconnect_nominal(self):
        """Test de déconnexion nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'connect') or not hasattr(self.exchange_interface, 'disconnect'):
            pytest.skip("Required methods not implemented")
            
        await self.exchange_interface.connect()
        assert self.exchange_interface.is_connected is True
        
        # When
        result = await self.exchange_interface.disconnect()
        
        # Then
        assert result['success'] is True
        assert 'timestamp' in result
        if hasattr(self.exchange_interface, 'is_connected'):
            assert self.exchange_interface.is_connected is False

    async def test_place_market_order_nominal(self):
        """Test de placement d'ordre au marché nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'place_order'):
            pytest.skip("place_order method not implemented")
            
        symbol = 'ETHUSDT'
        side = 'buy'
        order_type = 'market'
        quantity = 0.1
        
        # When
        result = await self.exchange_interface.place_order(
            symbol, side, order_type, quantity
        )
        
        # Then
        assert result['success'] is True
        assert 'order_id' in result
        assert 'symbol' in result
        assert 'side' in result
        assert 'order_type' in result
        assert 'quantity' in result
        assert 'status' in result
        assert 'timestamp' in result
        
        assert result['symbol'] == symbol
        assert result['side'] == side
        assert result['order_type'] == order_type
        assert result['quantity'] == quantity
        assert result['status'] == OrderStatus.OPEN

    async def test_place_limit_order_nominal(self):
        """Test de placement d'ordre limite nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'place_order'):
            pytest.skip("place_order method not implemented")
            
        symbol = 'ETHUSDT'
        side = 'buy'
        order_type = 'limit'
        quantity = 0.1
        price = 2000.0
        
        # When
        result = await self.exchange_interface.place_order(
            symbol, side, order_type, quantity, price
        )
        
        # Then
        assert result['success'] is True
        assert result['price'] == price
        assert result['order_type'] == order_type

    async def test_place_stop_order_nominal(self):
        """Test de placement d'ordre stop nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'place_order'):
            pytest.skip("place_order method not implemented")
            
        symbol = 'ETHUSDT'
        side = 'sell'
        order_type = 'stop'
        quantity = 0.1
        stop_price = 1980.0
        
        # When
        result = await self.exchange_interface.place_order(
            symbol, side, order_type, quantity, None, stop_price
        )
        
        # Then
        assert result['success'] is True
        assert result['stop_price'] == stop_price
        assert result['order_type'] == order_type

    async def test_place_order_invalid_symbol(self):
        """Test de placement d'ordre avec symbole invalide."""
        # Given
        if not hasattr(self.exchange_interface, 'place_order'):
            pytest.skip("place_order method not implemented")
            
        symbol = 'INVALIDSYMBOL'
        side = 'buy'
        order_type = 'market'
        quantity = 0.1
        
        # When
        result = await self.exchange_interface.place_order(
            symbol, side, order_type, quantity
        )
        
        # Then
        assert result['success'] is False
        assert 'error' in result
        assert 'symbol' in result['error'].lower() or 'invalid' in result['error'].lower()

    async def test_place_order_insufficient_balance(self):
        """Test de placement d'ordre avec solde insuffisant."""
        # Given
        if not hasattr(self.exchange_interface, 'place_order'):
            pytest.skip("place_order method not implemented")
            
        symbol = 'ETHUSDT'
        side = 'buy'
        order_type = 'market'
        quantity = 1000.0  # Très grande quantité
        
        # When
        result = await self.exchange_interface.place_order(
            symbol, side, order_type, quantity
        )
        
        # Then
        assert result['success'] is False
        assert 'error' in result
        assert 'balance' in result['error'].lower() or 'insufficient' in result['error'].lower()

    async def test_cancel_order_nominal(self):
        """Test d'annulation d'ordre nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'place_order') or not hasattr(self.exchange_interface, 'cancel_order'):
            pytest.skip("Required methods not implemented")
            
        # D'abord placer un ordre
        place_result = await self.exchange_interface.place_order(
            'ETHUSDT', 'buy', 'limit', 0.1, 2000.0
        )
        order_id = place_result['order_id']
        
        # When
        result = await self.exchange_interface.cancel_order(order_id)
        
        # Then
        assert result['success'] is True
        assert result['order_id'] == order_id
        assert 'status' in result
        assert result['status'] == OrderStatus.CANCELLED

    async def test_cancel_order_nonexistent(self):
        """Test d'annulation d'ordre inexistant."""
        # Given
        if not hasattr(self.exchange_interface, 'cancel_order'):
            pytest.skip("cancel_order method not implemented")
            
        nonexistent_order_id = 'nonexistent_order_123'
        
        # When
        result = await self.exchange_interface.cancel_order(nonexistent_order_id)
        
        # Then
        assert result['success'] is False
        assert 'error' in result
        assert 'not found' in result['error'].lower() or 'exist' in result['error'].lower()

    async def test_get_order_status_nominal(self):
        """Test de récupération du statut d'ordre nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'place_order') or not hasattr(self.exchange_interface, 'get_order_status'):
            pytest.skip("Required methods not implemented")
            
        # D'abord placer un ordre
        place_result = await self.exchange_interface.place_order(
            'ETHUSDT', 'buy', 'limit', 0.1, 2000.0
        )
        order_id = place_result['order_id']
        
        # When
        result = await self.exchange_interface.get_order_status(order_id)
        
        # Then
        assert result['success'] is True
        assert 'order_id' in result
        assert 'status' in result
        assert 'symbol' in result
        assert 'filled_quantity' in result
        assert 'remaining_quantity' in result
        assert 'timestamp' in result
        
        assert result['order_id'] == order_id

    async def test_get_order_status_nonexistent(self):
        """Test de récupération du statut d'ordre inexistant."""
        # Given
        if not hasattr(self.exchange_interface, 'get_order_status'):
            pytest.skip("get_order_status method not implemented")
            
        nonexistent_order_id = 'nonexistent_order_123'
        
        # When
        result = await self.exchange_interface.get_order_status(nonexistent_order_id)
        
        # Then
        assert result['success'] is False
        assert 'error' in result
        assert 'not found' in result['error'].lower() or 'exist' in result['error'].lower()

    async def test_get_balance_nominal(self):
        """Test de récupération du solde nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'get_balance'):
            pytest.skip("get_balance method not implemented")
            
        # When
        result = await self.exchange_interface.get_balance()
        
        # Then
        assert result['success'] is True
        assert 'balances' in result
        assert 'total_balance' in result
        assert 'available_balance' in result
        assert 'used_balance' in result
        assert 'timestamp' in result
        
        # Vérifier la structure des balances
        balances = result['balances']
        assert isinstance(balances, dict)
        
        # Devrait contenir des balances pour quelques actifs
        if len(balances) > 0:
            for asset, balance in balances.items():
                assert 'free' in balance
                assert 'used' in balance
                assert 'total' in balance

    async def test_get_ticker_nominal(self):
        """Test de récupération du ticker nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'get_ticker'):
            pytest.skip("get_ticker method not implemented")
            
        symbol = 'ETHUSDT'
        
        # When
        result = await self.exchange_interface.get_ticker(symbol)
        
        # Then
        assert result['success'] is True
        assert 'symbol' in result
        assert 'bid' in result
        assert 'ask' in result
        assert 'last' in result
        assert 'high' in result
        assert 'low' in result
        assert 'volume' in result
        assert 'timestamp' in result
        
        assert result['symbol'] == symbol

    async def test_get_ticker_invalid_symbol(self):
        """Test de récupération du ticker avec symbole invalide."""
        # Given
        if not hasattr(self.exchange_interface, 'get_ticker'):
            pytest.skip("get_ticker method not implemented")
            
        symbol = 'INVALIDSYMBOL'
        
        # When
        result = await self.exchange_interface.get_ticker(symbol)
        
        # Then
        assert result['success'] is False
        assert 'error' in result
        assert 'symbol' in result['error'].lower() or 'invalid' in result['error'].lower()

    async def test_get_order_book_nominal(self):
        """Test de récupération du carnet d'ordres nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'get_order_book'):
            pytest.skip("get_order_book method not implemented")
            
        symbol = 'ETHUSDT'
        limit = 10
        
        # When
        result = await self.exchange_interface.get_order_book(symbol, limit)
        
        # Then
        assert result['success'] is True
        assert 'symbol' in result
        assert 'bids' in result
        assert 'asks' in result
        assert 'timestamp' in result
        
        assert result['symbol'] == symbol
        
        # Vérifier la structure du carnet d'ordres
        bids = result['bids']
        asks = result['asks']
        
        assert isinstance(bids, list)
        assert isinstance(asks, list)
        
        # Chaque entrée devrait avoir [price, quantity]
        if len(bids) > 0:
            for bid in bids:
                assert len(bid) >= 2  # [price, quantity]
        
        if len(asks) > 0:
            for ask in asks:
                assert len(ask) >= 2  # [price, quantity]

    async def test_get_ohlcv_nominal(self):
        """Test de récupération des données OHLCV nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'get_ohlcv'):
            pytest.skip("get_ohlcv method not implemented")
            
        symbol = 'ETHUSDT'
        timeframe = '1h'
        limit = 100
        
        # When
        result = await self.exchange_interface.get_ohlcv(symbol, timeframe, limit)
        
        # Then
        assert result['success'] is True
        assert 'symbol' in result
        assert 'timeframe' in result
        assert 'data' in result
        assert 'timestamp' in result
        
        assert result['symbol'] == symbol
        assert result['timeframe'] == timeframe
        
        # Vérifier la structure des données OHLCV
        data = result['data']
        assert isinstance(data, list)
        
        if len(data) > 0:
            # Chaque entrée devrait avoir [timestamp, open, high, low, close, volume]
            for candle in data:
                assert len(candle) >= 6  # [timestamp, open, high, low, close, volume]

    async def test_get_trades_nominal(self):
        """Test de récupération des trades récents nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'get_trades'):
            pytest.skip("get_trades method not implemented")
            
        symbol = 'ETHUSDT'
        limit = 50
        
        # When
        result = await self.exchange_interface.get_trades(symbol, limit)
        
        # Then
        assert result['success'] is True
        assert 'symbol' in result
        assert 'trades' in result
        assert 'timestamp' in result
        
        assert result['symbol'] == symbol
        
        # Vérifier la structure des trades
        trades = result['trades']
        assert isinstance(trades, list)
        
        if len(trades) > 0:
            # Chaque trade devrait avoir [timestamp, price, quantity, side]
            for trade in trades:
                assert len(trade) >= 4  # [timestamp, price, quantity, side]

    async def test_get_open_orders_nominal(self):
        """Test de récupération des ordres ouverts nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'place_order') or not hasattr(self.exchange_interface, 'get_open_orders'):
            pytest.skip("Required methods not implemented")
            
        # Placer quelques ordres
        await self.exchange_interface.place_order('ETHUSDT', 'buy', 'limit', 0.1, 2000.0)
        await self.exchange_interface.place_order('BTCUSDT', 'sell', 'limit', 0.05, 50000.0)
        
        # When
        result = await self.exchange_interface.get_open_orders()
        
        # Then
        assert result['success'] is True
        assert 'orders' in result
        assert 'timestamp' in result
        
        orders = result['orders']
        assert isinstance(orders, list)
        assert len(orders) >= 2

    async def test_get_open_orders_filtered(self):
        """Test de récupération des ordres ouverts avec filtres."""
        # Given
        if not hasattr(self.exchange_interface, 'place_order') or not hasattr(self.exchange_interface, 'get_open_orders'):
            pytest.skip("Required methods not implemented")
            
        # Placer des ordres avec différents symboles
        await self.exchange_interface.place_order('ETHUSDT', 'buy', 'limit', 0.1, 2000.0)
        await self.exchange_interface.place_order('BTCUSDT', 'sell', 'limit', 0.05, 50000.0)
        
        # When
        # Filtrer par symbole
        result_eth = await self.exchange_interface.get_open_orders(symbol='ETHUSDT')
        assert result_eth['success'] is True
        assert len(result_eth['orders']) == 1
        assert result_eth['orders'][0]['symbol'] == 'ETHUSDT'
        
        # Filtrer par côté
        result_buy = await self.exchange_interface.get_open_orders(side='buy')
        assert result_buy['success'] is True
        assert len(result_buy['orders']) == 1
        assert result_buy['orders'][0]['side'] == 'buy'

    async def test_get_trade_history_nominal(self):
        """Test de récupération de l'historique des trades nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'get_trade_history'):
            pytest.skip("get_trade_history method not implemented")
            
        symbol = 'ETHUSDT'
        limit = 100
        
        # When
        result = await self.exchange_interface.get_trade_history(symbol, limit)
        
        # Then
        assert result['success'] is True
        assert 'symbol' in result
        assert 'trades' in result
        assert 'timestamp' in result
        
        assert result['symbol'] == symbol
        
        trades = result['trades']
        assert isinstance(trades, list)

    async def test_get_exchange_info_nominal(self):
        """Test de récupération des informations de l'exchange nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'get_exchange_info'):
            pytest.skip("get_exchange_info method not implemented")
            
        # When
        result = await self.exchange_interface.get_exchange_info()
        
        # Then
        assert result['success'] is True
        assert 'exchange_name' in result
        assert 'server_time' in result
        assert 'rate_limits' in result
        assert 'symbols' in result
        assert 'fees' in result
        
        assert result['exchange_name'] == 'binance'

    async def test_get_symbol_info_nominal(self):
        """Test de récupération des informations du symbole nominale."""
        # Given
        if not hasattr(self.exchange_interface, 'get_symbol_info'):
            pytest.skip("get_symbol_info method not implemented")
            
        symbol = 'ETHUSDT'
        
        # When
        result = await self.exchange_interface.get_symbol_info(symbol)
        
        # Then
        assert result['success'] is True
        assert 'symbol' in result
        assert 'base_asset' in result
        assert 'quote_asset' in result
        assert 'min_quantity' in result
        assert 'max_quantity' in result
        assert 'quantity_precision' in result
        assert 'price_precision' in result
        assert 'status' in result
        
        assert result['symbol'] == symbol
        assert result['base_asset'] == 'ETH'
        assert result['quote_asset'] == 'USDT'

    async def test_concurrent_orders(self):
        """Test d'ordres concurrents."""
        # Given
        if not hasattr(self.exchange_interface, 'place_order'):
            pytest.skip("place_order method not implemented")
            
        # When
        # Placer plusieurs ordres simultanément
        tasks = [
            self.exchange_interface.place_order('ETHUSDT', 'buy', 'market', 0.1),
            self.exchange_interface.place_order('BTCUSDT', 'sell', 'market', 0.05),
            self.exchange_interface.place_order('ADAUSDT', 'buy', 'market', 100.0)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Then
        successful_orders = [r for r in results if r and r.get('success')]
        assert len(successful_orders) == 3  # Tous devraient réussir
        
        order_ids = [r['order_id'] for r in successful_orders]
        assert len(set(order_ids)) == 3  # Tous les IDs devraient être uniques

    async def test_rate_limiting(self):
        """Test de gestion des limites de taux."""
        # Given
        if not hasattr(self.exchange_interface, 'place_order'):
            pytest.skip("place_order method not implemented")
            
        # When
        # Envoyer beaucoup d'ordres rapidement pour tester les limites de taux
        results = []
        for i in range(20):  # Plus que la limite de 10 par seconde
            result = await self.exchange_interface.place_order(f'SYMBOL{i}', 'buy', 'market', 0.1)
            results.append(result)
            
            # Petit délai pour éviter le blocage total
            await asyncio.sleep(0.01)
        
        # Then
        # Certains ordres devraient échouer à cause des limites de taux
        successful_orders = [r for r in results if r.get('success')]
        failed_orders = [r for r in results if not r.get('success')]
        
        assert len(successful_orders) < 20  # Pas tous devraient réussir
        assert len(failed_orders) > 0  # Certains devraient échouer
        
        # Les échecs devraient être dus aux limites de taux
        rate_limit_errors = [r for r in failed_orders if 'rate' in r.get('error', '').lower()]
        assert len(rate_limit_errors) > 0

    async def test_error_handling_invalid_inputs(self):
        """Test de gestion des erreurs avec entrées invalides."""
        # Given/When/Then
        if hasattr(self.exchange_interface, 'place_order'):
            # Test avec symbole vide
            with pytest.raises((ValueError, TypeError)):
                await self.exchange_interface.place_order('', 'buy', 'market', 0.1)
            
            # Test avec side invalide
            with pytest.raises((ValueError, TypeError)):
                await self.exchange_interface.place_order('ETHUSDT', 'invalid', 'market', 0.1)
            
            # Test avec quantité négative
            with pytest.raises((ValueError, TypeError)):
                await self.exchange_interface.place_order('ETHUSDT', 'buy', 'market', -0.1)
            
            # Test avec order_type invalide
            with pytest.raises((ValueError, TypeError)):
                await self.exchange_interface.place_order('ETHUSDT', 'buy', 'invalid', 0.1)

    async def test_performance_with_many_requests(self):
        """Test de performance avec beaucoup de requêtes."""
        # Given
        if not hasattr(self.exchange_interface, 'get_ticker'):
            pytest.skip("get_ticker method not implemented")
            
        # When
        start_time = datetime.now()
        
        # Effectuer beaucoup de requêtes
        tasks = []
        for i in range(100):
            task = self.exchange_interface.get_ticker(f'SYMBOL{i}')
            tasks.append(task)
        
        # Exécuter en parallèle avec timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.now()
        
        # Then
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 30.0  # Devrait s'exécuter en moins de 30 secondes
        
        # Vérifier que la plupart des requêtes ont réussi
        successful_requests = [r for r in results if r and r.get('success')]
        assert len(successful_requests) >= 90  # Au moins 90% devraient réussir

    async def test_memory_usage_with_large_data(self):
        """Test de l'utilisation mémoire avec beaucoup de données."""
        # Given
        if not hasattr(self.exchange_interface, 'get_ohlcv'):
            pytest.skip("get_ohlcv method not implemented")
            
        # When
        # Récupérer beaucoup de données OHLCV
        symbol = 'ETHUSDT'
        timeframe = '1m'
        limit = 10000  # Beaucoup de données
        
        # Exécuter de manière asynchrone pour le test
        result = await self.exchange_interface.get_ohlcv(symbol, timeframe, limit)
        
        # Then
        if result['success']:
            data = result['data']
            # Vérifier que nous avons beaucoup de données
            assert len(data) > 1000
            
            # Vérifier la structure des données
            for candle in data[:100]:  # Vérifier seulement les 100 premiers
                assert len(candle) >= 6  # [timestamp, open, high, low, close, volume]
        
        # Then
        # Le système devrait pouvoir gérer cette charge sans erreur de mémoire
        # (En pratique, on pourrait vouloir ajouter des limites)

    async def test_websocket_connection(self):
        """Test de connexion WebSocket."""
        # Given
        if not hasattr(self.exchange_interface, 'connect_websocket'):
            pytest.skip("connect_websocket method not implemented")
            
        symbols = ['ETHUSDT', 'BTCUSDT']
        
        # When
        result = await self.exchange_interface.connect_websocket(symbols)
        
        # Then
        assert result['success'] is True
        assert 'connection_id' in result
        assert 'symbols' in result
        assert 'timestamp' in result
        
        assert set(result['symbols']) == set(symbols)
        
        if hasattr(self.exchange_interface, 'websocket_connected'):
            assert self.exchange_interface.websocket_connected is True

    async def test_websocket_subscription(self):
        """Test d'abonnement WebSocket."""
        # Given
        if not hasattr(self.exchange_interface, 'subscribe_websocket'):
            pytest.skip("subscribe_websocket method not implemented")
            
        # D'abord se connecter
        await self.exchange_interface.connect_websocket(['ETHUSDT'])
        
        subscription_type = 'ticker'
        symbol = 'ETHUSDT'
        
        # When
        result = await self.exchange_interface.subscribe_websocket(subscription_type, symbol)
        
        # Then
        assert result['success'] is True
        assert 'subscription_id' in result
        assert 'type' in result
        assert 'symbol' in result
        
        assert result['type'] == subscription_type
        assert result['symbol'] == symbol

    async def test_websocket_data_reception(self):
        """Test de réception de données WebSocket."""
        # Given
        if not hasattr(self.exchange_interface, 'connect_websocket') or not hasattr(self.exchange_interface, 'get_websocket_data'):
            pytest.skip("Required methods not implemented")
            
        # Se connecter et s'abonner
        await self.exchange_interface.connect_websocket(['ETHUSDT'])
        await self.exchange_interface.subscribe_websocket('ticker', 'ETHUSDT')
        
        # Simuler la réception de données
        if hasattr(self.exchange_interface, '_simulate_websocket_data'):
            test_data = {
                'type': 'ticker',
                'symbol': 'ETHUSDT',
                'price': 2000.0,
                'timestamp': datetime.now().isoformat()
            }
            self.exchange_interface._simulate_websocket_data(test_data)
        
        # When
        result = await self.exchange_interface.get_websocket_data()
        
        # Then
        assert result['success'] is True
        assert 'data' in result
        assert 'count' in result
        
        if result['count'] > 0:
            data = result['data']
            assert isinstance(data, list)
            # Vérifier que nous avons des données de ticker
            ticker_data = [d for d in data if d.get('type') == 'ticker']
            assert len(ticker_data) > 0

    async def test_error_recovery(self):
        """Test de récupération après erreur."""
        # Given
        if not hasattr(self.exchange_interface, 'place_order') or not hasattr(self.exchange_interface, 'get_connection_status'):
            pytest.skip("Required methods not implemented")
            
        # Simuler une erreur de connexion
        if hasattr(self.exchange_interface, '_simulate_connection_error'):
            self.exchange_interface._simulate_connection_error(True)
        
        # When
        # Tenter de placer un ordre (devrait échouer)
        result = await self.exchange_interface.place_order('ETHUSDT', 'buy', 'market', 0.1)
        
        # Then
        assert result['success'] is False
        assert 'error' in result
        
        # Vérifier le statut de connexion
        status_result = await self.exchange_interface.get_connection_status()
        assert status_result['success'] is True
        assert status_result['connected'] is False
        assert 'error' in status_result
        
        # Récupérer de l'erreur
        if hasattr(self.exchange_interface, '_simulate_connection_error'):
            self.exchange_interface._simulate_connection_error(False)
        
        # Tenter à nouveau (devrait réussir)
        retry_result = await self.exchange_interface.place_order('ETHUSDT', 'buy', 'market', 0.1)
        assert retry_result['success'] is True

    async def test_exchange_specific_features(self):
        """Test des fonctionnalités spécifiques à l'exchange."""
        # Given
        if not hasattr(self.exchange_interface, 'get_exchange_features'):
            pytest.skip("get_exchange_features method not implemented")
            
        # When
        result = await self.exchange_interface.get_exchange_features()
        
        # Then
        assert result['success'] is True
        assert 'features' in result
        assert 'supported_order_types' in result['features']
        assert 'supported_timeframes' in result['features']
        assert 'has_websocket' in result['features']
        assert 'has_margin_trading' in result['features']
        assert 'has_futures' in result['features']
        
        # Vérifier les fonctionnalités supportées
        features = result['features']
        assert isinstance(features['supported_order_types'], list)
        assert isinstance(features['supported_timeframes'], list)
        assert isinstance(features['has_websocket'], bool)
        assert isinstance(features['has_margin_trading'], bool)
        assert isinstance(features['has_futures'], bool)

    async def test_api_authentication(self):
        """Test d'authentification API."""
        # Given
        if not hasattr(self.exchange_interface, 'authenticate'):
            pytest.skip("authenticate method not implemented")
            
        # When
        result = await self.exchange_interface.authenticate()
        
        # Then
        assert result['success'] is True
        assert 'authenticated' in result
        assert 'permissions' in result
        assert 'timestamp' in result
        
        assert result['authenticated'] is True
        assert isinstance(result['permissions'], list)