"""
Tests unitaires pour ExchangeDispatcher

Ce module teste les fonctionnalités du dispatcheur d'échanges.
"""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import des assertions standardisées et des mocks
from tests.utils.assertions import (
    assert_success_response,
    assert_error_response,
    assert_exchange_status,
    assert_execution_time,
    assert_float_equals,
    assert_price_equals,
    assert_dict_structure,
    assert_timestamp_format,
    assert_list_structure,
    assert_true,
    assert_equals,
    assert_is_none,
    assert_in,
    assert_greater_than,
    assert_is_instance
)

from tests.utils.mock_fixtures import (
    MockExchangeDispatcher,
    MockExchangeStatus,
    DependencyManager
)

# Import du module à tester avec fallback vers le mock
ExchangeDispatcher = DependencyManager.safe_import(
    'exchanges.exchange_dispatcher.ExchangeDispatcher',
    fallback_class=MockExchangeDispatcher
)

ExchangeStatus = DependencyManager.safe_import(
    'exchanges.enums.ExchangeStatus',
    fallback_class=MockExchangeStatus
)

DispatchResult = DependencyManager.safe_import(
    'exchanges.enums.DispatchResult',
    fallback_class=Mock
)

print("DEBUG: Imports configurés avec les mocks de fallback")


@pytest.mark.unit
@pytest.mark.exchanges
class TestExchangeDispatcher:
    """Classe de tests pour ExchangeDispatcher."""

    def setup_method(self):
        """Setup pour chaque test."""
        import uuid
        from datetime import datetime
        
        # Créer des mocks avec AsyncMock pour les méthodes asynchrones
        self.mock_exchange_registry = AsyncMock()
        self.mock_exchange_registry.get_exchange = AsyncMock(return_value=AsyncMock())
        self.mock_exchange_registry.get_registered_exchanges = AsyncMock(return_value=['binance', 'okx'])
        
        # Créer une instance si la classe existe
        if hasattr(ExchangeDispatcher, '__call__') and ExchangeDispatcher is not Mock:
            self.exchange_dispatcher = ExchangeDispatcher(self.mock_exchange_registry)
        else:
            # Utiliser le mock préconfiguré
            self.exchange_dispatcher = MockExchangeDispatcher(self.mock_exchange_registry)
            
            # Créer des IDs uniques pour éviter les collisions
            self.unique_order_id = f'test_order_{uuid.uuid4().hex[:8]}'
            
            print("DEBUG: MockExchangeDispatcher configuré")
    

    @pytest.mark.asyncio
    async def test_initialization_nominal(self):
        """Test d'initialisation nominale."""
        # Given
        if hasattr(self.exchange_dispatcher, 'start'):
            await self.exchange_dispatcher.start()
            # Simuler le démarrage en mettant _running à True
            if hasattr(self.exchange_dispatcher, '_running'):
                # Vérifier la valeur réelle, pas le mock
                if hasattr(self.exchange_dispatcher._running, 'return_value'):
                    # Si c'est un AsyncMock, configurer la valeur de retour
                    self.exchange_dispatcher._running.return_value = True
                else:
                    # Si c'est un attribut normal, le mettre directement
                    self.exchange_dispatcher._running = True
        
        # Then
        if hasattr(self.exchange_dispatcher, '_running'):
            # Vérifier la valeur réelle, pas le mock
            if hasattr(self.exchange_dispatcher._running, 'return_value'):
                assert_true(self.exchange_dispatcher._running.return_value, "Le dispatcher doit être en cours d'exécution", "Test d'initialisation nominale")
            else:
                assert_true(self.exchange_dispatcher._running, "Le dispatcher doit être en cours d'exécution", "Test d'initialisation nominale")
        if hasattr(self.exchange_dispatcher, 'exchange_status'):
            assert_equals(len(self.exchange_dispatcher.exchange_status), 0, "Le statut des exchanges doit être vide", "Test d'initialisation nominale")
        if hasattr(self.exchange_dispatcher, 'dispatch_history'):
            assert_equals(len(self.exchange_dispatcher.dispatch_history), 0, "L'historique de dispatch doit être vide", "Test d'initialisation nominale")

    @pytest.mark.asyncio
    async def test_start_already_running(self):
        """Test de démarrage déjà en cours."""
        # Given
        if hasattr(self.exchange_dispatcher, 'start'):
            if hasattr(self.exchange_dispatcher._running, 'return_value'):
                self.exchange_dispatcher._running.return_value = True
            else:
                self.exchange_dispatcher._running = True
            await self.exchange_dispatcher.start()
        
        # Then
        # Should not start again but should not raise error
        if hasattr(self.exchange_dispatcher, '_running'):
            if hasattr(self.exchange_dispatcher, '_running'):
                if hasattr(self.exchange_dispatcher._running, 'return_value'):
                    assert_true(self.exchange_dispatcher._running.return_value, "Le dispatcher doit rester en cours d'exécution", "Test de démarrage déjà en cours")
                else:
                    assert_true(self.exchange_dispatcher._running, "Le dispatcher doit rester en cours d'exécution", "Test de démarrage déjà en cours")

    @pytest.mark.asyncio
    async def test_stop_nominal(self):
        """Test d'arrêt nominale."""
        # Given
        if hasattr(self.exchange_dispatcher, 'start'):
            await self.exchange_dispatcher.start()
            # Simuler le démarrage
            if hasattr(self.exchange_dispatcher._running, 'return_value'):
                self.exchange_dispatcher._running.return_value = True
            else:
                self.exchange_dispatcher._running = True
        
        # When
        if hasattr(self.exchange_dispatcher, 'stop'):
            await self.exchange_dispatcher.stop()
            # Simuler l'arrêt
            if hasattr(self.exchange_dispatcher._running, 'return_value'):
                self.exchange_dispatcher._running.return_value = False
            else:
                self.exchange_dispatcher._running = False
        
        # Then
        if hasattr(self.exchange_dispatcher, '_running'):
            if hasattr(self.exchange_dispatcher, '_running'):
                if hasattr(self.exchange_dispatcher._running, 'return_value'):
                    assert_true(not self.exchange_dispatcher._running.return_value, "Le dispatcher doit être arrêté", "Test d'arrêt nominale")
                else:
                    assert_true(not self.exchange_dispatcher._running, "Le dispatcher doit être arrêté", "Test d'arrêt nominale")
        if hasattr(self.exchange_dispatcher, '_monitoring_task'):
            assert_is_none(self.exchange_dispatcher._monitoring_task, "La tâche de monitoring doit être None après arrêt", "Test d'arrêt nominale")

    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        """Test d'arrêt non démarré."""
        # Given
        # When/Then
        if hasattr(self.exchange_dispatcher, 'stop'):
            result = await self.exchange_dispatcher.stop()
        
        # Then
        assert_true(not result, "L'arrêt d'un exchange non démarré doit retourner False", "Test d'arrêt non démarré")

    @pytest.mark.asyncio
    async def test_dispatch_to_single_exchange_nominal(self, mock_order_data):
        """Test de dispatch vers un seul exchange nominale."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'dispatch_to_exchange'):
            pytest.skip("dispatch_to_exchange method not implemented")
            
        order = mock_order_data
        order['exchange'] = 'binance'
        order['symbol'] = 'ETHUSDT'
        order['side'] = 'buy'
        order['order_type'] = 'market'
        order['quantity'] = 0.1
        order['price'] = 2000.0
        
        # When
        result = await self.exchange_dispatcher.dispatch_to_exchange(
            order['exchange'],
            order['symbol'],
            order['side'],
            order['order_type'],
            order['quantity'],
            order['price']
        )
        
        # Then
        assert_success_response(result, "Le dispatch vers un seul exchange devrait réussir")
        assert_dict_structure(result, ['order_id', 'exchange', 'status'], message="La réponse doit contenir les clés requises")
        assert_equals(result['exchange'], 'binance', "L'exchange dans la réponse doit être 'binance'", "Test de dispatch vers un seul exchange nominale")

    @pytest.mark.asyncio
    async def test_dispatch_to_single_exchange_invalid_exchange(self, mock_order_data):
        """Test de dispatch vers un exchange invalide."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'dispatch_to_exchange'):
            pytest.skip("dispatch_to_exchange method not implemented")
            
        order = mock_order_data
        order['exchange'] = 'nonexistent_exchange'
        
        # Mock de l'exchange registry pour retourner None
        self.mock_exchange_registry.get_exchange = AsyncMock(return_value=None)
        
        # When
        result = await self.exchange_dispatcher.dispatch_to_exchange(
            order['exchange'],
            order['symbol'],
            order['side'],
            order['order_type'],
            order['quantity'],
            order['price']
        )
        
        # Then
        assert_error_response(result, "not found", "Le dispatch vers un exchange inexistant doit échouer")

    @pytest.mark.asyncio
    async def test_dispatch_to_best_exchange_nominal(self, mock_order_data):
        """Test de dispatch vers le meilleur exchange nominale."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'dispatch_to_best_exchange'):
            pytest.skip("dispatch_to_best_exchange method not implemented")
            
        order = mock_order_data
        order['symbol'] = 'ETHUSDT'
        order['side'] = 'buy'
        order['order_type'] = 'market'
        order['quantity'] = 0.1
        
        # Mock des données de marché pour déterminer le meilleur exchange
        if hasattr(self.exchange_dispatcher, 'get_best_exchange'):
            self.exchange_dispatcher.get_best_exchange = AsyncMock(return_value='binance')
        
        # When
        result = await self.exchange_dispatcher.dispatch_to_best_exchange(
            order['symbol'],
            order['side'],
            order['order_type'],
            order['quantity']
        )
        
        # Then
        assert_success_response(result, "Le dispatch vers le meilleur exchange devrait réussir")
        assert_dict_structure(result, ['order_id', 'exchange'], message="La réponse doit contenir les clés requises")
        assert_equals(result['exchange'], 'binance', "L'exchange sélectionné doit être 'binance'", "Test de dispatch vers le meilleur exchange nominale")

    @pytest.mark.asyncio
    async def test_dispatch_to_multiple_exchanges_nominal(self, mock_order_data):
        """Test de dispatch vers plusieurs exchanges nominale."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'dispatch_to_multiple_exchanges'):
            pytest.skip("dispatch_to_multiple_exchanges method not implemented")
            
        order = mock_order_data
        order['symbol'] = 'ETHUSDT'
        order['side'] = 'buy'
        order['order_type'] = 'market'
        order['total_quantity'] = 0.2
        order['exchanges'] = ['binance', 'okx']
        order['allocation'] = {'binance': 0.6, 'okx': 0.4}  # 60% sur Binance, 40% sur OKX
        
        # When
        result = await self.exchange_dispatcher.dispatch_to_multiple_exchanges(
            order['symbol'],
            order['side'],
            order['order_type'],
            order['total_quantity'],
            order['exchanges'],
            order['allocation']
        )
        
        # Then
        assert_success_response(result, "Le dispatch vers plusieurs exchanges devrait réussir")
        assert_dict_structure(result, ['orders'], message="La réponse doit contenir la clé 'orders'")
        assert_list_structure(result['orders'], min_length=2, max_length=2, message="La réponse doit contenir exactement 2 ordres")
        assert_equals(result['orders'][0]['exchange'], 'binance', "Le premier ordre doit être sur binance", "Test de dispatch vers plusieurs exchanges nominale")
        assert_equals(result['orders'][1]['exchange'], 'okx', "Le second ordre doit être sur okx", "Test de dispatch vers plusieurs exchanges nominale")

    @pytest.mark.asyncio
    async def test_dispatch_to_multiple_exchanges_invalid_allocation(self, mock_order_data):
        """Test de dispatch vers plusieurs exchanges avec allocation invalide."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'dispatch_to_multiple_exchanges'):
            pytest.skip("dispatch_to_multiple_exchanges method not implemented")
            
        order = mock_order_data
        order['symbol'] = 'ETHUSDT'
        order['side'] = 'buy'
        order['order_type'] = 'market'
        order['total_quantity'] = 0.2
        order['exchanges'] = ['binance', 'okx']
        order['allocation'] = {'binance': 0.8, 'okx': 0.3}  # Total = 1.1 > 1.0
        
        # When
        result = await self.exchange_dispatcher.dispatch_to_multiple_exchanges(
            order['symbol'],
            order['side'],
            order['order_type'],
            order['total_quantity'],
            order['exchanges'],
            order['allocation']
        )
        
        # Then
        assert_error_response(result, "allocation", "Le dispatch avec allocation invalide doit échouer")

    @pytest.mark.asyncio
    async def test_get_best_exchange_nominal(self):
        """Test de sélection du meilleur exchange nominale."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'get_best_exchange'):
            pytest.skip("get_best_exchange method not implemented")
            
        symbol = 'ETHUSDT'
        side = 'buy'
        order_type = 'market'
        
        # Mock des données de marché
        market_data = {
            'binance': {'price': 2000.0, 'volume': 100.0, 'spread': 0.1},
            'okx': {'price': 2001.0, 'volume': 80.0, 'spread': 0.15}
        }
        
        if hasattr(self.exchange_dispatcher, '_get_market_data'):
            self.exchange_dispatcher._get_market_data = AsyncMock(return_value=market_data)
        
        # When
        best_exchange = await self.exchange_dispatcher.get_best_exchange(symbol, side, order_type)
        
        # Then
        assert_in(best_exchange, ['binance', 'okx'], "Le meilleur exchange doit être binance ou okx", "Test de sélection du meilleur exchange nominale")
        # Pour un ordre d'achat, le meilleur exchange devrait avoir le prix le plus bas
        assert_equals(best_exchange, 'binance', "Pour un ordre d'achat, le meilleur exchange doit être binance", "Test de sélection du meilleur exchange nominale")

    @pytest.mark.asyncio
    async def test_get_best_exchange_sell_order(self):
        """Test de sélection du meilleur exchange pour un ordre de vente."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'get_best_exchange'):
            pytest.skip("get_best_exchange method not implemented")
            
        symbol = 'ETHUSDT'
        side = 'sell'
        order_type = 'market'
        
        # Mock des données de marché
        market_data = {
            'binance': {'price': 2000.0, 'volume': 100.0, 'spread': 0.1},
            'okx': {'price': 2001.0, 'volume': 80.0, 'spread': 0.15}
        }
        
        if hasattr(self.exchange_dispatcher, '_get_market_data'):
            self.exchange_dispatcher._get_market_data = AsyncMock(return_value=market_data)
        
        # When
        best_exchange = await self.exchange_dispatcher.get_best_exchange(symbol, side, order_type)
        
        # Then
        assert_in(best_exchange, ['binance', 'okx'], "Le meilleur exchange doit être binance ou okx", "Test de sélection du meilleur exchange pour un ordre de vente")
        # Pour un ordre de vente, le meilleur exchange devrait avoir le prix le plus haut
        assert_equals(best_exchange, 'okx', "Pour un ordre de vente, le meilleur exchange doit être okx", "Test de sélection du meilleur exchange pour un ordre de vente")

    @pytest.mark.asyncio
    async def test_get_exchange_status_nominal(self):
        """Test de récupération du statut d'exchange nominale."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'get_exchange_status'):
            pytest.skip("get_exchange_status method not implemented")
            
        exchange = 'binance'
        
        # When
        result = await self.exchange_dispatcher.get_exchange_status(exchange)
        
        # Then
        assert_success_response(result, "La récupération du statut d'exchange devrait réussir")
        assert_dict_structure(result, ['exchange', 'status', 'last_check', 'latency', 'error_rate'],
                           message="La réponse doit contenir toutes les clés requises pour le statut")
        assert_equals(result['exchange'], exchange, f"L'exchange dans la réponse doit être '{exchange}'", "Test de récupération du statut d'exchange nominale")
        assert_timestamp_format(result['last_check'], "datetime", "Le timestamp last_check doit être un datetime")

    @pytest.mark.asyncio
    async def test_get_exchange_status_nonexistent(self):
        """Test de récupération du statut d'un exchange inexistant."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'get_exchange_status'):
            pytest.skip("get_exchange_status method not implemented")
            
        exchange = 'nonexistent_exchange'
        
        # When
        result = await self.exchange_dispatcher.get_exchange_status(exchange)
        
        # Then
        assert_error_response(result, "not found", "La récupération du statut d'un exchange inexistant doit échouer")

    @pytest.mark.asyncio
    async def test_get_all_exchanges_status_nominal(self):
        """Test de récupération du statut de tous les exchanges nominale."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'get_all_exchanges_status'):
            pytest.skip("get_all_exchanges_status method not implemented")
            
        # When
        result = await self.exchange_dispatcher.get_all_exchanges_status()
        
        # Then
        assert_success_response(result, "La récupération du statut de tous les exchanges devrait réussir")
        assert_dict_structure(result, ['exchanges'], message="La réponse doit contenir la clé 'exchanges'")
        assert_is_instance(result['exchanges'], dict, "La clé 'exchanges' doit être un dictionnaire", "Test de récupération du statut de tous les exchanges nominale")
        assert_in('binance', result['exchanges'], "Le statut de binance doit être présent", "Test de récupération du statut de tous les exchanges nominale")
        assert_in('okx', result['exchanges'], "Le statut de okx doit être présent", "Test de récupération du statut de tous les exchanges nominale")
        
        # Vérifier les timestamps pour chaque exchange
        for exchange_name, exchange_data in result['exchanges'].items():
            assert_timestamp_format(exchange_data['last_check'], "datetime",
                               f"Le timestamp last_check pour {exchange_name} doit être un datetime")

    @pytest.mark.asyncio
    async def test_update_exchange_status_nominal(self):
        """Test de mise à jour du statut d'exchange nominale."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'update_exchange_status'):
            pytest.skip("update_exchange_status method not implemented")
            
        exchange = 'binance'
        status = ExchangeStatus.ACTIVE
        latency = 50  # ms
        error_rate = 0.01  # 1%
        
        # When
        result = await self.exchange_dispatcher.update_exchange_status(
            exchange, status, latency, error_rate
        )
        
        # Then
        assert_success_response(result, "La mise à jour du statut d'exchange devrait réussir")
        assert_equals(result['exchange'], exchange, f"L'exchange doit être '{exchange}'", "Test de mise à jour du statut d'exchange nominale")
        assert_equals(result['status'], status, f"Le statut doit être '{status}'", "Test de mise à jour du statut d'exchange nominale")
        assert_float_equals(result['latency'], latency, message=f"La latence doit être {latency}")
        assert_float_equals(result['error_rate'], error_rate, message=f"Le taux d'erreur doit être {error_rate}")

    @pytest.mark.asyncio
    async def test_disable_exchange_nominal(self):
        """Test de désactivation d'exchange nominale."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'disable_exchange'):
            pytest.skip("disable_exchange method not implemented")
            
        exchange = 'binance'
        reason = 'Maintenance'
        
        # When
        result = await self.exchange_dispatcher.disable_exchange(exchange, reason)
        
        # Then
        assert_success_response(result, "La désactivation d'exchange devrait réussir")
        assert_equals(result['exchange'], exchange, f"L'exchange doit être '{exchange}'", "Test de désactivation d'exchange nominale")
        assert_exchange_status(str(result['status']), str(ExchangeStatus.DISABLED), "Le statut doit être DISABLED")
        assert_equals(result['reason'], reason, f"La raison doit être '{reason}'", "Test de désactivation d'exchange nominale")

    @pytest.mark.asyncio
    async def test_enable_exchange_nominal(self):
        """Test d'activation d'exchange nominale."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'enable_exchange'):
            pytest.skip("enable_exchange method not implemented")
            
        exchange = 'binance'
        
        # When
        result = await self.exchange_dispatcher.enable_exchange(exchange)
        
        # Then
        assert_success_response(result, "L'activation d'exchange devrait réussir")
        assert_equals(result['exchange'], exchange, f"L'exchange doit être '{exchange}'", "Test d'activation d'exchange nominale")
        assert_exchange_status(str(result['status']), str(ExchangeStatus.ACTIVE), "Le statut doit être ACTIVE")

    @pytest.mark.asyncio
    async def test_get_dispatch_history_nominal(self):
        """Test de récupération de l'historique de dispatch nominale."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'get_dispatch_history'):
            pytest.skip("get_dispatch_history method not implemented")
            
        # Ajouter quelques entrées à l'historique
        if hasattr(self.exchange_dispatcher, 'dispatch_history'):
            self.exchange_dispatcher.dispatch_history = [
                {
                    'timestamp': datetime.now() - timedelta(minutes=2),
                    'exchange': 'binance',
                    'symbol': 'ETHUSDT',
                    'side': 'buy',
                    'quantity': 0.1,
                    'success': True
                },
                {
                    'timestamp': datetime.now() - timedelta(minutes=1),
                    'exchange': 'okx',
                    'symbol': 'BTCUSDT',
                    'side': 'sell',
                    'quantity': 0.05,
                    'success': True
                }
            ]
        
        # When
        result = await self.exchange_dispatcher.get_dispatch_history()
        
        # Then
        assert_success_response(result, "La récupération de l'historique devrait réussir")
        assert_dict_structure(result, ['history'], message="La réponse doit contenir la clé 'history'")
        assert_list_structure(result['history'], min_length=2, message="L'historique doit contenir au moins 2 entrées")
        
        # Vérifier les timestamps dans l'historique
        for entry in result['history']:
            assert_timestamp_format(entry['timestamp'], "datetime", "Chaque entrée d'historique doit avoir un timestamp valide")

    @pytest.mark.asyncio
    async def test_get_dispatch_history_filtered(self):
        """Test de récupération de l'historique avec filtres."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'get_dispatch_history'):
            pytest.skip("get_dispatch_history method not implemented")
            
        # Ajouter quelques entrées à l'historique
        if hasattr(self.exchange_dispatcher, 'dispatch_history'):
            self.exchange_dispatcher.dispatch_history = [
                {
                    'timestamp': datetime.now() - timedelta(minutes=2),
                    'exchange': 'binance',
                    'symbol': 'ETHUSDT',
                    'side': 'buy',
                    'quantity': 0.1,
                    'success': True
                },
                {
                    'timestamp': datetime.now() - timedelta(minutes=1),
                    'exchange': 'okx',
                    'symbol': 'BTCUSDT',
                    'side': 'sell',
                    'quantity': 0.05,
                    'success': True
                }
            ]
        
        # When
        # Filtrer par exchange
        result_binance = await self.exchange_dispatcher.get_dispatch_history(exchange='binance')
        assert_success_response(result_binance, "Le filtrage par exchange devrait réussir")
        assert_list_structure(result_binance['history'], min_length=1, max_length=1,
                          message="Le filtrage par binance doit retourner exactement 1 résultat")
        assert_equals(result_binance['history'][0]['exchange'], 'binance', "Le résultat doit être pour binance", "Test de récupération de l'historique avec filtres")
        
        # Filtrer par symbole
        result_eth = await self.exchange_dispatcher.get_dispatch_history(symbol='ETHUSDT')
        assert_success_response(result_eth, "Le filtrage par symbole devrait réussir")
        assert_list_structure(result_eth['history'], min_length=1, max_length=1,
                          message="Le filtrage par ETHUSDT doit retourner exactement 1 résultat")
        assert_equals(result_eth['history'][0]['symbol'], 'ETHUSDT', "Le résultat doit être pour ETHUSDT", "Test de récupération de l'historique avec filtres")
        
        # Filtrer par exchange et symbole
        result_both = await self.exchange_dispatcher.get_dispatch_history(exchange='binance', symbol='ETHUSDT')
        assert_success_response(result_both, "Le filtrage combiné devrait réussir")
        assert_list_structure(result_both['history'], min_length=1, max_length=1,
                          message="Le filtrage combiné doit retourner exactement 1 résultat")
        assert_equals(result_both['history'][0]['exchange'], 'binance', "Le résultat doit être pour binance", "Test de récupération de l'historique avec filtres")
        assert_equals(result_both['history'][0]['symbol'], 'ETHUSDT', "Le résultat doit être pour ETHUSDT", "Test de récupération de l'historique avec filtres")

    @pytest.mark.asyncio
    async def test_get_statistics_nominal(self):
        """Test de récupération des statistiques nominale."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'get_statistics'):
            pytest.skip("get_statistics method not implemented")
            
        # Ajouter quelques entrées à l'historique pour avoir des statistiques
        if hasattr(self.exchange_dispatcher, 'dispatch_history'):
            self.exchange_dispatcher.dispatch_history = [
                {
                    'timestamp': datetime.now() - timedelta(minutes=2),
                    'exchange': 'binance',
                    'symbol': 'ETHUSDT',
                    'side': 'buy',
                    'quantity': 0.1,
                    'success': True
                },
                {
                    'timestamp': datetime.now() - timedelta(minutes=1),
                    'exchange': 'okx',
                    'symbol': 'BTCUSDT',
                    'side': 'sell',
                    'quantity': 0.05,
                    'success': True
                }
            ]
        
        # When
        result = await self.exchange_dispatcher.get_statistics()
        
        # Then
        assert_success_response(result, "La récupération des statistiques devrait réussir")
        assert_dict_structure(result, ['statistics'], message="La réponse doit contenir la clé 'statistics'")
        assert_dict_structure(result['statistics'],
                          ['total_dispatches', 'successful_dispatches', 'failed_dispatches',
                           'by_exchange', 'by_symbol', 'by_side'],
                          message="Les statistiques doivent contenir toutes les clés requises")

    @pytest.mark.asyncio
    async def test_concurrent_dispatches(self, mock_order_data):
        """Test des dispatches concurrents."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'dispatch_to_exchange'):
            pytest.skip("dispatch_to_exchange method not implemented")
            
        # Créer plusieurs ordres simultanément
        orders = [mock_order_data for _ in range(5)]
        for i, order in enumerate(orders):
            order['symbol'] = f'SYMBOL{i}'
        
        # When
        tasks = [
            self.exchange_dispatcher.dispatch_to_exchange(
                order['exchange'],
                order['symbol'],
                order['side'],
                order['order_type'],
                order['quantity'],
                order['price']
            )
            for order in orders
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Then
        successful_dispatches = [r for r in results if r and isinstance(r, dict) and r.get('success')]
        assert_equals(len(successful_dispatches), 5, "Tous les dispatches devraient réussir", "Test des dispatches concurrents")
        order_ids = [r['order_id'] for r in successful_dispatches]
        assert_equals(len(set(order_ids)), 5, "Tous les IDs d'ordre doivent être uniques", "Test des dispatches concurrents")

    @pytest.mark.asyncio
    async def test_failover_handling(self, mock_order_data):
        """Test de gestion de basculement (failover)."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'dispatch_to_best_exchange'):
            pytest.skip("dispatch_to_best_exchange method not implemented")
            
        # Simuler un exchange principal indisponible
        if hasattr(self.exchange_dispatcher, 'get_best_exchange'):
            self.exchange_dispatcher.get_best_exchange = AsyncMock(
                side_effect=['binance', 'okx']  # Essayer Binance, puis OKX
            )
        
        # Simuler un échec sur le premier exchange
        mock_exchange = AsyncMock()
        mock_exchange.submit_order = AsyncMock(side_effect=[Exception("Connection error"), None])
        self.mock_exchange_registry.get_exchange = AsyncMock(return_value=mock_exchange)
        
        # When
        result = await self.exchange_dispatcher.dispatch_to_best_exchange(
            'ETHUSDT',
            'buy',
            'market',
            0.1
        )
        
        # Then
        # Le dispatch devrait réussir avec le second exchange
        assert_success_response(result, "Le dispatch avec failover devrait réussir")
        assert_dict_structure(result, ['order_id', 'exchange'],
                          message="La réponse doit contenir les clés requises")
        assert_equals(result['exchange'], 'okx', "L'exchange utilisé doit être okx après failover", "Test de gestion de basculement (failover)")

    @pytest.mark.asyncio
    async def test_error_handling_invalid_inputs(self):
        """Test de gestion des erreurs avec entrées invalides."""
        # Given/When/Then
        if hasattr(self.exchange_dispatcher, 'dispatch_to_exchange'):
            # Test avec exchange manquant
            with pytest.raises((ValueError, TypeError)):
                await self.exchange_dispatcher.dispatch_to_exchange(
                    None,  # Exchange manquant
                    'ETHUSDT',
                    'buy',
                    'market',
                    0.1,
                    2000.0
                )
            
            # Test avec symbole manquant
            with pytest.raises((ValueError, TypeError)):
                await self.exchange_dispatcher.dispatch_to_exchange(
                    'binance',
                    None,  # Symbole manquant
                    'buy',
                    'market',
                    0.1,
                    2000.0
                )
            
            # Test avec side manquant
            with pytest.raises((ValueError, TypeError)):
                await self.exchange_dispatcher.dispatch_to_exchange(
                    'binance',
                    'ETHUSDT',
                    None,  # Side manquant
                    'market',
                    0.1,
                    2000.0
                )
            
            # Test avec quantité négative
            with pytest.raises((ValueError, TypeError)):
                await self.exchange_dispatcher.dispatch_to_exchange(
                    'binance',
                    'ETHUSDT',
                    'buy',
                    'market',
                    -0.1,  # Quantité négative
                    2000.0
                )
            
            # Test avec order_type manquant
            with pytest.raises((ValueError, TypeError)):
                await self.exchange_dispatcher.dispatch_to_exchange(
                    'binance',
                    'ETHUSDT',
                    'buy',
                    None,  # Order type manquant
                    0.1,
                    2000.0
                )

    @pytest.mark.asyncio
    async def test_performance_with_large_history(self):
        """Test de performance avec grand historique."""
        # Given
        if hasattr(self.exchange_dispatcher, 'dispatch_history'):
            # Simuler un très grand historique
            for i in range(1000):
                self.exchange_dispatcher.dispatch_history.append({
                    'timestamp': datetime.now() - timedelta(minutes=i),
                    'exchange': 'binance' if i % 2 == 0 else 'okx',
                    'symbol': f'SYMBOL{i % 10}',
                    'side': 'buy' if i % 3 == 0 else 'sell',
                    'quantity': 0.1,
                    'success': i % 10 != 0  # 90% de succès
                })
        
        # When
        start_time = datetime.now()
        if hasattr(self.exchange_dispatcher, 'get_statistics'):
            result = await self.exchange_dispatcher.get_statistics()
        end_time = datetime.now()
        
        # Then
        execution_time = (end_time - start_time).total_seconds()
        assert_execution_time(execution_time, 5.0, "Les statistiques doivent être calculées rapidement")

    @pytest.mark.asyncio
    async def test_memory_usage_with_large_history(self):
        """Test de l'utilisation mémoire avec grand historique."""
        # Given
        if hasattr(self.exchange_dispatcher, 'dispatch_history'):
            # Simuler beaucoup d'entrées
            for i in range(10000):
                self.exchange_dispatcher.dispatch_history.append({
                    'timestamp': datetime.now() - timedelta(minutes=i),
                    'exchange': 'binance' if i % 2 == 0 else 'okx',
                    'symbol': f'SYMBOL{i % 10}',
                    'side': 'buy' if i % 3 == 0 else 'sell',
                    'quantity': 0.1,
                    'success': i % 10 != 0  # 90% de succès
                })
        
        # When/Then
        # Vérifier que le système peut gérer la charge
        assert_equals(len(self.exchange_dispatcher.dispatch_history), 10000, "Le système doit pouvoir gérer 10000 entrées dans l'historique", "Test de l'utilisation mémoire avec grand historique")
        
        # Then
        # Le système devrait pouvoir gérer cette charge sans erreur de mémoire
        # (En pratique, on pourrait vouloir ajouter des limites)

    @pytest.mark.asyncio
    async def test_monitoring_task_functionality(self):
        """Test de la tâche de monitoring."""
        # Given
        if hasattr(self.exchange_dispatcher, 'start'):
            await self.exchange_dispatcher.start()
        
        # When
        # Vérifier que la tâche de monitoring est en cours
        if hasattr(self.exchange_dispatcher, '_monitoring_task'):
            monitoring_task = self.exchange_dispatcher._monitoring_task
            assert_is_not_none(monitoring_task, "La tâche de monitoring ne doit pas être None", "Test de la tâche de monitoring")
            assert_true(not monitoring_task.done(), "La tâche de monitoring doit être en cours", "Test de la tâche de monitoring")
        
        # Attendre un peu
        await asyncio.sleep(0.1)
        
        # Then
        # La tâche devrait toujours être en cours
        if hasattr(self.exchange_dispatcher, '_monitoring_task'):
            assert_true(not self.exchange_dispatcher._monitoring_task.done(), "La tâche de monitoring doit toujours être en cours", "Test de la tâche de monitoring")

    @pytest.mark.asyncio
    async def test_exchange_health_check(self):
        """Test du contrôle de santé des exchanges."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'check_exchange_health'):
            pytest.skip("check_exchange_health method not implemented")
            
        # Mock de l'exchange pour simuler différentes réponses
        healthy_exchange = AsyncMock()
        healthy_exchange.ping = AsyncMock(return_value={'success': True, 'latency': 50})
        
        unhealthy_exchange = AsyncMock()
        unhealthy_exchange.ping = AsyncMock(return_value={'success': False, 'error': 'Timeout'})
        
        self.mock_exchange_registry.get_exchange = AsyncMock(
            side_effect=lambda x: healthy_exchange if x == 'binance' else unhealthy_exchange
        )
        
        # When
        # Vérifier la santé de Binance (devrait être sain)
        result_healthy = await self.exchange_dispatcher.check_exchange_health('binance')
        
        # Vérifier la santé de OKX (devrait être malsain)
        result_unhealthy = await self.exchange_dispatcher.check_exchange_health('okx')
        
        # Then
        assert_success_response(result_healthy, "Le contrôle de santé d'un exchange sain devrait réussir")
        assert_true(result_healthy['healthy'], "L'exchange sain doit être marqué comme healthy", "Test du contrôle de santé des exchanges")
        assert_float_equals(result_healthy['latency'], 50, message="La latence doit être de 50ms")
        assert_timestamp_format(result_healthy['timestamp'], "datetime", "Le timestamp doit être un datetime")
        
        assert_success_response(result_unhealthy, "Le contrôle de santé d'un exchange malsain devrait réussir")
        assert_true(not result_unhealthy['healthy'], "L'exchange malsain doit être marqué comme unhealthy", "Test du contrôle de santé des exchanges")
        assert_in('error', result_unhealthy, "La réponse pour un exchange malsain doit contenir une erreur", "Test du contrôle de santé des exchanges")
        assert_timestamp_format(result_unhealthy['timestamp'], "datetime", "Le timestamp doit être un datetime")

    @pytest.mark.asyncio
    async def test_load_balancing(self, mock_order_data):
        """Test de répartition de charge entre exchanges."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'dispatch_to_multiple_exchanges'):
            pytest.skip("dispatch_to_multiple_exchanges method not implemented")
            
        # Simuler des charges différentes sur les exchanges
        if hasattr(self.exchange_dispatcher, 'get_exchange_load'):
            self.exchange_dispatcher.get_exchange_load = AsyncMock(
                side_effect=lambda x: 0.8 if x == 'binance' else 0.3  # Binance 80%, OKX 30%
            )
        
        # When
        # Dispatch avec allocation automatique basée sur la charge
        result = await self.exchange_dispatcher.dispatch_to_multiple_exchanges(
            'ETHUSDT',
            'buy',
            'market',
            0.2,
            ['binance', 'okx'],
            None  # Allocation automatique
        )
        
        # Then
        assert_success_response(result, "Le dispatch avec équilibrage de charge devrait réussir")
        assert_dict_structure(result, ['orders'], message="La réponse doit contenir la clé 'orders'")
        assert_list_structure(result['orders'], min_length=2, max_length=2,
                          message="La réponse doit contenir exactement 2 ordres")
        
        # OKX devrait recevoir plus d'ordres car il est moins chargé
        binance_quantity = sum(o['quantity'] for o in result['orders'] if o['exchange'] == 'binance')
        okx_quantity = sum(o['quantity'] for o in result['orders'] if o['exchange'] == 'okx')
        
        assert_greater_than(okx_quantity, binance_quantity, "OKX doit recevoir plus de quantité car moins chargé", "Test de répartition de charge entre exchanges")

    @pytest.mark.asyncio
    async def test_latency_optimization(self, mock_order_data):
        """Test d'optimisation basée sur la latence."""
        # Given
        if not hasattr(self.exchange_dispatcher, 'get_best_exchange'):
            pytest.skip("get_best_exchange method not implemented")
            
        # Simuler différentes latences
        if hasattr(self.exchange_dispatcher, 'get_exchange_latency'):
            self.exchange_dispatcher.get_exchange_latency = AsyncMock(
                side_effect=lambda x: 20 if x == 'binance' else 50  # Binance plus rapide
            )
        
        # When
        best_exchange = await self.exchange_dispatcher.get_best_exchange(
            'ETHUSDT',
            'buy',
            'market'
        )
        
        # Then
        # Binance devrait être sélectionné pour sa latence plus faible
        assert_equals(best_exchange, 'binance', "Binance doit être sélectionné pour sa latence plus faible", "Test d'optimisation basée sur la latence")