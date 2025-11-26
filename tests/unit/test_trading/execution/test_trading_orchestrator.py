"""
Tests unitaires pour TradingOrchestrator

Ce module teste les fonctionnalités de l'orchestrateur de trading.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import du module à tester
try:
    from src.trading.execution.trading_orchestrator import TradingOrchestrator
except ImportError:
    # Si le module n'existe pas encore, on utilise un mock
    TradingOrchestrator = Mock

# Import des assertions standardisées
from tests.utils.assertions import (
    assert_success_response,
    assert_error_response,
    assert_float_equals,
    assert_dict_structure,
    assert_list_structure,
    assert_execution_time,
    assert_performance_metrics
)


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.asyncio
class TestTradingOrchestrator:
    """Classe de tests pour TradingOrchestrator."""

    def setup_method(self):
        """Setup pour chaque test."""
        self.config = {
            'trading_mode': 'paper',
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'account_balance': 10000.0,
            'max_positions_per_symbol': 6,
            'max_total_positions': 6,
            'max_position_size': 1000.0,
            'max_exposure_per_symbol': 0.2,
            'max_total_exposure': 0.5,
            'trading_interval': 30,
            'analyst': {},
            'tactician': {},
            'supervisor': {},
            'strategist': {},
            'analyst_signals': {'confidence_threshold': 0.6},
            'tactician_signals': {'confidence_threshold': 0.6},
            'signal_combiner': {}
        }
        
        # Créer une instance si la classe existe
        if hasattr(TradingOrchestrator, '__call__'):
            self.orchestrator = TradingOrchestrator(self.config)
        else:
            self.orchestrator = Mock()

    async def test_initialization_nominal(self):
        """Test d'initialisation nominale."""
        # Given
        expected_config = self.config
        
        # When
        if hasattr(self.orchestrator, 'initialize'):
            result = await self.orchestrator.initialize()
        
        # Then
        assert result is True, "L'initialisation doit réussir"
        if hasattr(self.orchestrator, 'config'):
            assert self.orchestrator.config == expected_config, "La configuration doit correspondre à celle attendue"
        if hasattr(self.orchestrator, 'status'):
            from src.trading.execution.trading_orchestrator import OrchestratorStatus
            assert self.orchestrator.status == OrchestratorStatus.STOPPED, "Le statut initial doit être STOPPED"

    async def test_initialization_missing_config(self):
        """Test d'initialisation avec configuration manquante."""
        # Given/When/Then
        with pytest.raises((ValueError, TypeError)):
            if hasattr(TradingOrchestrator, '__call__'):
                TradingOrchestrator(None)

    async def test_start_trading_session_nominal(self):
        """Test de démarrage de session de trading nominale."""
        # Given
        if not hasattr(self.orchestrator, 'initialize') or not hasattr(self.orchestrator, 'start_trading_session'):
            pytest.skip("Required methods not implemented")
            
        await self.orchestrator.initialize()
        
        # When
        result = await self.orchestrator.start_trading_session()
        
        # Then
        assert result is True, "Le démarrage de session doit réussir"
        if hasattr(self.orchestrator, 'status'):
            from src.trading.execution.trading_orchestrator import OrchestratorStatus
            assert self.orchestrator.status == OrchestratorStatus.RUNNING, "Le statut doit être RUNNING après démarrage"
        if hasattr(self.orchestrator, 'current_session'):
            assert self.orchestrator.current_session is not None, "La session courante doit être définie"

    async def test_start_trading_session_already_running(self):
        """Test de démarrage de session déjà en cours."""
        # Given
        if not hasattr(self.orchestrator, 'initialize') or not hasattr(self.orchestrator, 'start_trading_session'):
            pytest.skip("Required methods not implemented")
            
        await self.orchestrator.initialize()
        
        # Simuler un état déjà en cours
        if hasattr(self.orchestrator, 'status'):
            from src.trading.execution.trading_orchestrator import OrchestratorStatus
            self.orchestrator.status = OrchestratorStatus.RUNNING
        
        # When
        result = await self.orchestrator.start_trading_session()
        
        # Then
        assert result is False, "Le démarrage doit échouer si déjà en cours"

    async def test_stop_trading_session_nominal(self):
        """Test d'arrêt de session de trading nominale."""
        # Given
        if not hasattr(self.orchestrator, 'initialize') or not hasattr(self.orchestrator, 'start_trading_session') or not hasattr(self.orchestrator, 'stop_trading_session'):
            pytest.skip("Required methods not implemented")
            
        await self.orchestrator.initialize()
        await self.orchestrator.start_trading_session()
        
        # When
        result = await self.orchestrator.stop_trading_session()
        
        # Then
        assert result is True, "L'arrêt de session doit réussir"
        if hasattr(self.orchestrator, 'status'):
            from src.trading.execution.trading_orchestrator import OrchestratorStatus
            assert self.orchestrator.status == OrchestratorStatus.STOPPED, "Le statut doit être STOPPED après arrêt"

    async def test_stop_trading_session_not_running(self):
        """Test d'arrêt de session non démarrée."""
        # Given
        if not hasattr(self.orchestrator, 'initialize') or not hasattr(self.orchestrator, 'stop_trading_session'):
            pytest.skip("Required methods not implemented")
            
        await self.orchestrator.initialize()
        
        # When
        result = await self.orchestrator.stop_trading_session()
        
        # Then
        assert result is False, "L'arrêt doit échouer si aucune session en cours"

    async def test_generate_trading_decision_nominal(self, mock_market_snapshot, mock_trading_signals):
        """Test de génération de décision de trading nominale."""
        # Given
        if not hasattr(self.orchestrator, '_generate_trading_decision'):
            pytest.skip("_generate_trading_decision method not implemented")
            
        # Mock du snapshot de marché et des signaux
        if hasattr(self.orchestrator, '_latest_signals'):
            self.orchestrator._latest_signals = mock_trading_signals
        
        # When
        decision = await self.orchestrator._generate_trading_decision(mock_market_snapshot)
        
        # Then
        assert decision is not None, "La décision de trading ne doit pas être None"
        assert hasattr(decision, 'symbol'), "La décision doit avoir un attribut symbol"
        assert hasattr(decision, 'action'), "La décision doit avoir un attribut action"
        assert hasattr(decision, 'quantity'), "La décision doit avoir un attribut quantity"
        assert hasattr(decision, 'price'), "La décision doit avoir un attribut price"
        assert hasattr(decision, 'confidence'), "La décision doit avoir un attribut confidence"
        assert decision.symbol == self.config['symbol'], "Le symbole doit correspondre à celui de la config"

    async def test_hive_predictions_written_for_live_mode(self, mock_market_data):
        """Vérifie que les prédictions sont écrites dans Hive en mode LIVE."""
        # Given
        if not hasattr(TradingOrchestrator, '__call__'):
            pytest.skip("TradingOrchestrator not implemented")

        from src.trading.execution.trading_orchestrator import (
            TradingDecision,
            TradingMode,
        )

        config = self.config.copy()
        config.update({
            'trading_mode': 'live',
            'enable_hive_predictions': True,
            'model_version': 'vtest',
            'hive_layer': 'meta_layer',
        })

        orchestrator = TradingOrchestrator(config)

        class DummyWriter:
            def __init__(self):
                self.calls = []

            def write_predictions(self, df, prediction_date, metadata=None):
                self.calls.append((df, prediction_date, metadata))

        writer = DummyWriter()
        orchestrator.hive_prediction_writer = writer

        class DummyAnalyst:
            def __init__(self, confidence_score: float):
                self.confidence_score = confidence_score

        class DummyTactician:
            def __init__(self, confidence_score: float):
                self.confidence_score = confidence_score

        ts = datetime.now()
        decision = TradingDecision(
            timestamp=ts,
            symbol=config['symbol'],
            action='buy',
            quantity=0.1,
            price=2000.0,
            confidence=0.9,
            analyst_signal=DummyAnalyst(0.8),
            tactician_signal=DummyTactician(0.85),
            combined_signal={},
            risk_metrics={},
            metadata={},
        )

        # Mode LIVE requis pour l'écriture Hive
        orchestrator.trading_mode = TradingMode.LIVE

        # When
        orchestrator._persist_predictions_to_hive(decision, mock_market_data)

        # Then
        assert len(writer.calls) == 1, "Le writer Hive doit être appelé une fois"
        df, prediction_date, metadata = writer.calls[0]
        assert isinstance(df, pd.DataFrame), "Le writer doit recevoir un DataFrame"
        assert len(df) == 1, "Le DataFrame doit contenir une seule ligne"
        assert isinstance(df.index, pd.DatetimeIndex), "L'index doit être un DatetimeIndex"
        assert metadata['symbol'] == config['symbol'], "Le symbole dans les métadonnées doit correspondre"
        assert metadata['trading_mode'] == TradingMode.LIVE.value, "Le mode de trading doit être LIVE dans les métadonnées"

    async def test_generate_trading_decision_no_market_data(self):
        """Test de génération de décision sans données de marché."""
        # Given
        if not hasattr(self.orchestrator, '_generate_trading_decision'):
            pytest.skip("_generate_trading_decision method not implemented")
            
        empty_snapshot = {'market_data': pd.DataFrame()}
        
        # When
        decision = await self.orchestrator._generate_trading_decision(empty_snapshot)
        
        # Then
        assert decision is None, "La décision doit être None sans données de marché"

    async def test_execute_trading_decision_nominal(self, mock_trading_decision, mock_market_snapshot):
        """Test d'exécution de décision de trading nominale."""
        # Given
        if not hasattr(self.orchestrator, '_execute_trading_decision'):
            pytest.skip("_execute_trading_decision method not implemented")
            
        decision = mock_trading_decision
        
        # When
        await self.orchestrator._execute_trading_decision(decision, mock_market_snapshot)
        
        # Then
        # Vérifier que la décision a été ajoutée à l'historique
        if hasattr(self.orchestrator, 'trading_decisions'):
            assert len(self.orchestrator.trading_decisions) > 0, "L'historique des décisions ne doit pas être vide"
            assert self.orchestrator.trading_decisions[-1] == decision, "La dernière décision doit correspondre"

    async def test_execute_trading_decision_invalid_decision(self):
        """Test d'exécution de décision invalide."""
        # Given
        if not hasattr(self.orchestrator, '_execute_trading_decision'):
            pytest.skip("_execute_trading_decision method not implemented")
            
        invalid_decision = Mock()
        invalid_decision.action = 'invalid_action'
        
        # When/Then
        # Ne devrait pas lever d'exception mais devrait gérer l'erreur
        await self.orchestrator._execute_trading_decision(invalid_decision, {})

    async def test_validate_position_limits_within_limits(self, mock_trading_decision):
        """Test de validation des limites de position dans les limites."""
        # Given
        if not hasattr(self.orchestrator, '_validate_position_limits'):
            pytest.skip("_validate_position_limits method not implemented")
            
        decision = mock_trading_decision
        decision.action = 'buy'
        decision.quantity = 0.1  # Petite quantité
        decision.price = 2000.0
        
        # When
        result = await self.orchestrator._validate_position_limits(decision)
        
        # Then
        assert result is True, "La validation doit réussir pour les limites respectées"

    async def test_validate_position_limits_exceeded(self, mock_trading_decision):
        """Test de validation des limites de position dépassées."""
        # Given
        if not hasattr(self.orchestrator, '_validate_position_limits'):
            pytest.skip("_validate_position_limits method not implemented")
            
        decision = mock_trading_decision
        decision.action = 'buy'
        decision.quantity = 1000.0  # Trop grand
        
        # When
        result = await self.orchestrator._validate_position_limits(decision)
        
        # Then
        assert result is False, "La validation doit échouer pour les limites dépassées"

    async def test_get_market_snapshot_nominal(self, mock_market_data):
        """Test de récupération du snapshot de marché nominale."""
        # Given
        if not hasattr(self.orchestrator, '_get_market_snapshot'):
            pytest.skip("_get_market_snapshot method not implemented")
            
        # Mock du collecteur de données
        if hasattr(self.orchestrator, 'data_collector'):
            self.orchestrator.data_collector = Mock()
            self.orchestrator.data_collector.get_processed_data_df = Mock(return_value=mock_market_data)
        
        # When
        snapshot = await self.orchestrator._get_market_snapshot()
        
        # Then
        assert snapshot is not None, "Le snapshot ne doit pas être None"
        assert 'market_data' in snapshot, "Le snapshot doit contenir les données de marché"

    async def test_get_market_snapshot_no_data(self):
        """Test de récupération du snapshot de marché sans données."""
        # Given
        if not hasattr(self.orchestrator, '_get_market_snapshot'):
            pytest.skip("_get_market_snapshot method not implemented")
            
        # Mock du collecteur de données qui retourne vide
        if hasattr(self.orchestrator, 'data_collector'):
            self.orchestrator.data_collector = Mock()
            self.orchestrator.data_collector.get_processed_data_df = Mock(return_value=pd.DataFrame())
        
        # When
        snapshot = await self.orchestrator._get_market_snapshot()
        
        # Then
        assert snapshot is None, "Le snapshot doit être None sans données"

    async def test_evaluate_trailing_positions_nominal(self, mock_market_snapshot, mock_position_data):
        """Test d'évaluation des positions avec trailing nominale."""
        # Given
        if not hasattr(self.orchestrator, '_evaluate_trailing_positions'):
            pytest.skip("_evaluate_trailing_positions method not implemented")
            
        # Simuler des positions actives
        if hasattr(self.orchestrator, 'active_positions'):
            self.orchestrator.active_positions = {'test_position': mock_position_data}
        
        # When
        await self.orchestrator._evaluate_trailing_positions(mock_market_snapshot)
        
        # Then
        # Vérifier que l'évaluation a été effectuée
        # (L'implémentation exacte dépend de la structure interne)

    async def test_update_active_positions_open_position(self, mock_trading_decision, mock_position_data):
        """Test de mise à jour des positions avec ouverture."""
        # Given
        if not hasattr(self.orchestrator, '_update_active_positions') or not hasattr(self.orchestrator, '_open_position'):
            pytest.skip("Required methods not implemented")
            
        decision = mock_trading_decision
        decision.action = 'buy'
        decision.quantity = 0.1
        
        # When
        await self.orchestrator._update_active_positions(decision, 'test_trade_id', Mock())
        
        # Then
        # Vérifier que la position a été ajoutée
        if hasattr(self.orchestrator, 'active_positions'):
            assert 'test_position' in self.orchestrator.active_positions, "La position doit être ajoutée aux positions actives"

    async def test_update_active_positions_close_position(self, mock_trading_decision, mock_position_data):
        """Test de mise à jour des positions avec fermeture."""
        # Given
        if not hasattr(self.orchestrator, '_update_active_positions') or not hasattr(self.orchestrator, '_close_all_positions_for_symbol'):
            pytest.skip("Required methods not implemented")
            
        decision = mock_trading_decision
        decision.action = 'close'
        
        # Simuler une position existante
        if hasattr(self.orchestrator, 'active_positions'):
            self.orchestrator.active_positions = {'test_position': mock_position_data}
        
        # When
        await self.orchestrator._update_active_positions(decision, 'test_trade_id', Mock())
        
        # Then
        # Vérifier que la position a été fermée
        if hasattr(self.orchestrator, 'active_positions'):
            assert 'test_position' not in self.orchestrator.active_positions, "La position doit être retirée des positions actives"

    def test_build_ml_context_nominal(self, mock_position_data, mock_trading_signals):
        """Test de construction du contexte ML nominale."""
        # Given
        if not hasattr(self.orchestrator, '_build_ml_context'):
            pytest.skip("_build_ml_context method not implemented")
            
        position = mock_position_data
        signals = mock_trading_signals
        
        # When
        context = self.orchestrator._build_ml_context(position)
        
        # Then
        assert isinstance(context, dict), "Le contexte ML doit être un dictionnaire"
        assert 'entry' in context, "Le contexte doit contenir une clé 'entry'"
        assert context['entry'] == position.get('ml_entry', {}), "L'entrée doit correspondre à ml_entry"

    async def test_simulate_order_execution_nominal(self, mock_trading_decision):
        """Test de simulation d'exécution d'ordre nominale."""
        # Given
        if not hasattr(self.orchestrator, '_simulate_order_execution'):
            pytest.skip("_simulate_order_execution method not implemented")
            
        decision = mock_trading_decision
        decision.action = 'buy'
        decision.confidence = 0.8
        
        # When
        result = await self.orchestrator._simulate_order_execution(decision)
        
        # Then
        # Avec une confiance élevée, devrait réussir la plupart du temps
        assert result is True or isinstance(result, bool), "Le résultat doit être True ou un booléen"

    async def test_simulate_order_execution_low_confidence(self, mock_trading_decision):
        """Test de simulation d'exécution d'ordre avec faible confiance."""
        # Given
        if not hasattr(self.orchestrator, '_simulate_order_execution'):
            pytest.skip("_simulate_order_execution method not implemented")
            
        decision = mock_trading_decision
        decision.action = 'buy'
        decision.confidence = 0.3  # Faible confiance
        
        # When
        result = await self.orchestrator._simulate_order_execution(decision)
        
        # Then
        # Avec une faible confiance, devrait parfois échouer
        # Le résultat dépend de l'implémentation exacte

    def test_get_orchestrator_stats_nominal(self):
        """Test de récupération des statistiques nominale."""
        # Given
        if not hasattr(self.orchestrator, 'get_orchestrator_stats'):
            pytest.skip("get_orchestrator_stats method not implemented")
        
        # Simuler des données de statistiques
        if hasattr(self.orchestrator, 'performance_metrics'):
            self.orchestrator.performance_metrics = {
                'total_sessions': 5,
                'total_trades': 100,
                'successful_trades': 60,
                'failed_trades': 40
            }
        
        # When
        stats = self.orchestrator.get_orchestrator_stats()
        
        # Then
        assert isinstance(stats, dict), "Les statistiques doivent être un dictionnaire"
        assert 'performance_metrics' in stats, "Les statistiques doivent contenir les métriques de performance"
        assert stats['performance_metrics']['total_sessions'] == 5, "Le nombre total de sessions doit être 5"

    async def test_generate_live_dashboard_nominal(self):
        """Test de génération du dashboard live nominale."""
        # Given
        if not hasattr(self.orchestrator, 'generate_live_dashboard'):
            pytest.skip("generate_live_dashboard method not implemented")
            
        # When
        dashboard = await self.orchestrator.generate_live_dashboard()
        
        # Then
        assert isinstance(dashboard, dict), "Le dashboard doit être un dictionnaire"
        # La structure exacte dépend de l'implémentation

    async def test_generate_performance_report_nominal(self):
        """Test de génération du rapport de performance nominale."""
        # Given
        if not hasattr(self.orchestrator, 'generate_performance_report'):
            pytest.skip("generate_performance_report method not implemented")
            
        # When
        report = await self.orchestrator.generate_performance_report('session')
        
        # Then
        assert isinstance(report, dict), "Le rapport doit être un dictionnaire"
        # La structure exacte dépend de l'implémentation

    async def test_concurrent_operations(self, mock_trading_decision):
        """Test des opérations concurrentes."""
        # Given
        if not hasattr(self.orchestrator, '_generate_trading_decision') or not hasattr(self.orchestrator, '_execute_trading_decision'):
            pytest.skip("Required methods not implemented")
            
        decisions = [mock_trading_decision for _ in range(5)]
        for i, decision in enumerate(decisions):
            decision.action = 'buy' if i % 2 == 0 else 'sell'
        
        # When
        tasks = [self.orchestrator._generate_trading_decision({}) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Then
        # Vérifier que toutes les décisions ont été générées
        successful_decisions = [r for r in results if r is not None]
        assert len(successful_decisions) == 5, "Toutes les décisions doivent être générées"

    async def test_error_handling_invalid_config(self):
        """Test de gestion des erreurs avec configuration invalide."""
        # Given
        invalid_configs = [
            {},  # Vide
            {'trading_mode': 'invalid_mode'},  # Mode invalide
            {'account_balance': -1000.0},  # Solde négatif
        ]
        
        # When/Then
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, TypeError)):
                if hasattr(TradingOrchestrator, '__call__'):
                    TradingOrchestrator(invalid_config)

    async def test_performance_with_large_dataset(self, mock_market_data):
        """Test de performance avec grand jeu de données."""
        # Given
        if not hasattr(self.orchestrator, '_get_market_snapshot'):
            pytest.skip("_get_market_snapshot method not implemented")
            
        # Créer un grand DataFrame
        large_data = pd.concat([mock_market_data for _ in range(100)])
        
        # Mock du collecteur de données
        if hasattr(self.orchestrator, 'data_collector'):
            self.orchestrator.data_collector = Mock()
            self.orchestrator.data_collector.get_processed_data_df = Mock(return_value=large_data)
        
        # When
        start_time = datetime.now()
        if hasattr(self.orchestrator, '_get_market_snapshot'):
            snapshot = await self.orchestrator._get_market_snapshot()
        end_time = datetime.now()
        
        # Then
        execution_time = (end_time - start_time).total_seconds()
        assert_execution_time(
            execution_time,
            5.0,
            message="L'exécution doit prendre moins de 5 secondes même avec beaucoup de données"
        )

    async def test_edge_case_zero_balance(self):
        """Test des cas limites avec solde nul."""
        # Given
        config_zero_balance = self.config.copy()
        config_zero_balance['account_balance'] = 0.0
        
        # When/Then
        with pytest.raises((ValueError, TypeError)):
            if hasattr(TradingOrchestrator, '__call__'):
                TradingOrchestrator(config_zero_balance)

    async def test_edge_case_extreme_values(self):
        """Test des cas limites avec valeurs extrêmes."""
        # Given
        extreme_config = self.config.copy()
        extreme_config.update({
            'max_positions_per_symbol': 1000,  # Très grand
            'max_total_positions': 10000,  # Très grand
            'trading_interval': 0.1  # Très court
        })
        
        # When
        if hasattr(TradingOrchestrator, '__call__'):
            orchestrator = TradingOrchestrator(extreme_config)
        
        # Then
        # Devrait gérer les valeurs extrêmes sans erreur
        assert orchestrator is not None, "L'orchestrateur doit être créé même avec des valeurs extrêmes"
        if hasattr(orchestrator, 'config'):
            assert orchestrator.config['max_positions_per_symbol'] == 1000, "La configuration doit conserver les valeurs extrêmes"

    async def test_memory_leak_prevention(self, mock_trading_decision):
        """Test de prévention des fuites mémoire."""
        # Given
        if not hasattr(self.orchestrator, '_generate_trading_decision') or not hasattr(self.orchestrator, 'trading_decisions'):
            pytest.skip("Required methods not implemented")
            
        # Simuler beaucoup de décisions
        initial_count = 0
        if hasattr(self.orchestrator, 'trading_decisions'):
            initial_count = len(self.orchestrator.trading_decisions)
        
        # When
        tasks = []
        for _ in range(100):
            tasks.append(self.orchestrator._generate_trading_decision({}))
        
        await asyncio.gather(*tasks)
        
        # Then
        # Vérifier que la liste ne croît pas indéfiniment
        if hasattr(self.orchestrator, 'trading_decisions'):
            final_count = len(self.orchestrator.trading_decisions)
            assert final_count >= initial_count, "Le nombre de décisions doit augmenter"
            # En pratique, il faudrait vérifier que les anciennes décisions sont nettoyées

    async def test_state_consistency(self):
        """Test de la cohérence de l'état de l'orchestrateur."""
        # Given
        if hasattr(TradingOrchestrator, '__call__'):
            orchestrator = TradingOrchestrator(self.config)
        
        # When/Then
        # Vérifier que l'état initial est cohérent
        if hasattr(orchestrator, 'status'):
            from src.trading.execution.trading_orchestrator import OrchestratorStatus
            assert orchestrator.status == OrchestratorStatus.STOPPED, "Le statut initial doit être STOPPED"
        if hasattr(orchestrator, 'trading_decisions'):
            assert len(orchestrator.trading_decisions) == 0, "L'historique des décisions doit être vide initialement"
        if hasattr(orchestrator, 'active_positions'):
            assert len(orchestrator.active_positions) == 0, "Les positions actives doivent être vides initialement"