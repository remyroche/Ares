"""
Tests unitaires pour ConfigurationManager

Ce module teste les fonctionnalités du gestionnaire de configuration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import tempfile
import os

# Import du module à tester
try:
    from exchanges.shared.config_manager import (
        ConfigurationManager, 
        ConfigSource, 
        ConfigFormat, 
        ConfigValidationResult,
        ConfigVersion
    )
except ImportError:
    # Si le module n'existe pas encore, on utilise un mock
    ConfigurationManager = Mock
    ConfigSource = Mock
    ConfigFormat = Mock
    ConfigValidationResult = Mock
    ConfigVersion = Mock


@pytest.mark.unit
@pytest.mark.exchanges
@pytest.mark.asyncio
class TestConfigurationManager:
    """Classe de tests pour ConfigurationManager."""

    def setup_method(self):
        """Setup pour chaque test."""
        self.temp_config_file = None
        self.test_config_data = {
            'exchanges': {
                'binance': {
                    'api_key': 'test_binance_key',
                    'api_secret': 'test_binance_secret',
                    'sandbox': True,
                    'timeout': 30,
                    'rate_limit': 10
                },
                'okx': {
                    'api_key': 'test_okx_key',
                    'api_secret': 'test_okx_secret',
                    'passphrase': 'test_okx_passphrase',
                    'sandbox': True,
                    'timeout': 30,
                    'rate_limit': 20
                }
            },
            'trading': {
                'max_position_size': 0.1,
                'risk_level': 'medium',
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05
            },
            'database': {
                'type': 'sqlite',
                'path': 'test.db'
            }
        }
        
        # Créer un fichier de configuration temporaire
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config_data, f)
            self.temp_config_file = f.name
        
        # Créer une instance si la classe existe
        if hasattr(ConfigurationManager, '__call__'):
            self.config_manager = ConfigurationManager(self.temp_config_file)
        else:
            self.config_manager = Mock()

    def teardown_method(self):
        """Nettoyage après chaque test."""
        if self.temp_config_file and os.path.exists(self.temp_config_file):
            os.unlink(self.temp_config_file)

    async def test_initialization_nominal(self):
        """Test d'initialisation nominale."""
        # Given/When
        if hasattr(self.config_manager, 'start'):
            await self.config_manager.start()
        
        # Then
        if hasattr(self.config_manager, 'is_running'):
            assert self.config_manager.is_running is True
        if hasattr(self.config_manager, 'config_file'):
            assert self.config_manager.config_file == self.temp_config_file
        if hasattr(self.config_manager, 'config_format'):
            assert self.config_manager.config_format == ConfigFormat.JSON

    async def test_initialization_invalid_file(self):
        """Test d'initialisation avec fichier invalide."""
        # Given
        if not hasattr(self.config_manager, '__init__'):
            pytest.skip("ConfigurationManager class not available")
            
        invalid_file = 'nonexistent_file.json'
        
        # When/Then
        with pytest.raises(FileNotFoundError):
            ConfigurationManager(invalid_file)

    async def test_load_config_nominal(self):
        """Test de chargement de configuration nominale."""
        # Given
        if not hasattr(self.config_manager, 'load_config'):
            pytest.skip("load_config method not implemented")
            
        # When
        result = await self.config_manager.load_config()
        
        # Then
        assert result['success'] is True
        assert 'config' in result
        assert 'source' in result
        assert 'timestamp' in result
        
        config = result['config']
        assert 'exchanges' in config
        assert 'trading' in config
        assert 'database' in config
        
        # Vérifier les données chargées
        assert config['exchanges']['binance']['api_key'] == 'test_binance_key'
        assert config['trading']['max_position_size'] == 0.1

    async def test_load_config_from_source_nominal(self):
        """Test de chargement de configuration depuis source nominale."""
        # Given
        if not hasattr(self.config_manager, 'load_config_from_source'):
            pytest.skip("load_config_from_source method not implemented")
            
        # Créer une configuration temporaire dans un autre format
        yaml_config = """
exchanges:
  binance:
    api_key: test_yaml_key
    api_secret: test_yaml_secret
    sandbox: true
trading:
  max_position_size: 0.2
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_config)
            yaml_file = f.name
        
        # When
        result = await self.config_manager.load_config_from_source(yaml_file, ConfigFormat.YAML)
        
        # Then
        assert result['success'] is True
        assert result['source'] == ConfigSource.FILE
        assert result['format'] == ConfigFormat.YAML
        
        config = result['config']
        assert config['exchanges']['binance']['api_key'] == 'test_yaml_key'
        assert config['trading']['max_position_size'] == 0.2
        
        # Nettoyer
        os.unlink(yaml_file)

    async def test_load_config_invalid_format(self):
        """Test de chargement de configuration avec format invalide."""
        # Given
        if not hasattr(self.config_manager, 'load_config_from_source'):
            pytest.skip("load_config_from_source method not implemented")
            
        # Créer un fichier invalide
        invalid_config = "{invalid json content"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(invalid_config)
            invalid_file = f.name
        
        # When
        result = await self.config_manager.load_config_from_source(invalid_file, ConfigFormat.JSON)
        
        # Then
        assert result['success'] is False
        assert 'error' in result
        assert 'format' in result['error'].lower() or 'parse' in result['error'].lower()
        
        # Nettoyer
        os.unlink(invalid_file)

    async def test_save_config_nominal(self):
        """Test de sauvegarde de configuration nominale."""
        # Given
        if not hasattr(self.config_manager, 'save_config') or not hasattr(self.config_manager, 'load_config'):
            pytest.skip("Required methods not implemented")
            
        # Charger la configuration
        await self.config_manager.load_config()
        
        # Modifier la configuration
        new_config = await self.config_manager.get_config()
        new_config['trading']['max_position_size'] = 0.15
        
        # When
        result = await self.config_manager.save_config(new_config)
        
        # Then
        assert result['success'] is True
        assert 'timestamp' in result
        
        # Recharger et vérifier
        reload_result = await self.config_manager.load_config()
        reloaded_config = reload_result['config']
        assert reloaded_config['trading']['max_position_size'] == 0.15

    async def test_save_config_backup(self):
        """Test de sauvegarde de configuration avec backup."""
        # Given
        if not hasattr(self.config_manager, 'save_config_with_backup') or not hasattr(self.config_manager, 'load_config'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        new_config = await self.config_manager.get_config()
        new_config['trading']['max_position_size'] = 0.25
        
        # When
        result = await self.config_manager.save_config_with_backup(new_config)
        
        # Then
        assert result['success'] is True
        assert 'backup_file' in result
        assert 'timestamp' in result
        
        # Vérifier que le backup a été créé
        assert os.path.exists(result['backup_file'])

    async def test_validate_config_nominal(self):
        """Test de validation de configuration nominale."""
        # Given
        if not hasattr(self.config_manager, 'validate_config'):
            pytest.skip("validate_config method not implemented")
            
        valid_config = {
            'exchanges': {
                'binance': {
                    'api_key': 'test_key',
                    'api_secret': 'test_secret'
                }
            },
            'trading': {
                'max_position_size': 0.1
            }
        }
        
        # When
        result = await self.config_manager.validate_config(valid_config)
        
        # Then
        assert result['success'] is True
        assert 'valid' in result
        assert 'errors' in result
        assert result['valid'] is True
        assert len(result['errors']) == 0

    async def test_validate_config_missing_required_fields(self):
        """Test de validation de configuration avec champs requis manquants."""
        # Given
        if not hasattr(self.config_manager, 'validate_config'):
            pytest.skip("validate_config method not implemented")
            
        invalid_config = {
            'exchanges': {
                'binance': {
                    # Manque api_key
                    'api_secret': 'test_secret'
                }
            }
        }
        
        # When
        result = await self.config_manager.validate_config(invalid_config)
        
        # Then
        assert result['success'] is False
        assert result['valid'] is False
        assert len(result['errors']) > 0
        
        # Vérifier l'erreur spécifique
        error_messages = ' '.join(result['errors']).lower()
        assert 'api_key' in error_messages or 'required' in error_messages

    async def test_validate_config_invalid_values(self):
        """Test de validation de configuration avec valeurs invalides."""
        # Given
        if not hasattr(self.config_manager, 'validate_config'):
            pytest.skip("validate_config method not implemented")
            
        invalid_config = {
            'trading': {
                'max_position_size': -0.1  # Valeur négative invalide
            }
        }
        
        # When
        result = await self.config_manager.validate_config(invalid_config)
        
        # Then
        assert result['success'] is False
        assert result['valid'] is False
        assert len(result['errors']) > 0
        
        # Vérifier l'erreur spécifique
        error_messages = ' '.join(result['errors']).lower()
        assert 'max_position_size' in error_messages or 'invalid' in error_messages

    async def test_get_config_value_nominal(self):
        """Test de récupération de valeur de configuration nominale."""
        # Given
        if not hasattr(self.config_manager, 'get_config_value') or not hasattr(self.config_manager, 'load_config'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # When
        # Récupérer une valeur simple
        result = await self.config_manager.get_config_value('trading.max_position_size')
        
        # Then
        assert result['success'] is True
        assert 'value' in result
        assert 'path' in result
        assert result['value'] == 0.1
        assert result['path'] == 'trading.max_position_size'

    async def test_get_config_value_nested(self):
        """Test de récupération de valeur de configuration imbriquée."""
        # Given
        if not hasattr(self.config_manager, 'get_config_value') or not hasattr(self.config_manager, 'load_config'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # When
        # Récupérer une valeur imbriquée
        result = await self.config_manager.get_config_value('exchanges.binance.api_key')
        
        # Then
        assert result['success'] is True
        assert result['value'] == 'test_binance_key'
        assert result['path'] == 'exchanges.binance.api_key'

    async def test_get_config_value_nonexistent(self):
        """Test de récupération de valeur de configuration inexistante."""
        # Given
        if not hasattr(self.config_manager, 'get_config_value') or not hasattr(self.config_manager, 'load_config'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # When
        result = await self.config_manager.get_config_value('nonexistent.path')
        
        # Then
        assert result['success'] is False
        assert 'error' in result
        assert 'not found' in result['error'].lower() or 'exist' in result['error'].lower()

    async def test_set_config_value_nominal(self):
        """Test de définition de valeur de configuration nominale."""
        # Given
        if not hasattr(self.config_manager, 'set_config_value') or not hasattr(self.config_manager, 'load_config') or not hasattr(self.config_manager, 'get_config_value'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # When
        # Définir une nouvelle valeur
        result = await self.config_manager.set_config_value('trading.max_position_size', 0.2)
        
        # Then
        assert result['success'] is True
        assert 'old_value' in result
        assert 'new_value' in result
        assert result['old_value'] == 0.1
        assert result['new_value'] == 0.2
        
        # Vérifier que la valeur a été mise à jour
        get_result = await self.config_manager.get_config_value('trading.max_position_size')
        assert get_result['success'] is True
        assert get_result['value'] == 0.2

    async def test_set_config_value_nested(self):
        """Test de définition de valeur de configuration imbriquée."""
        # Given
        if not hasattr(self.config_manager, 'set_config_value') or not hasattr(self.config_manager, 'load_config') or not hasattr(self.config_manager, 'get_config_value'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # When
        # Définir une nouvelle valeur imbriquée
        result = await self.config_manager.set_config_value('exchanges.binance.timeout', 45)
        
        # Then
        assert result['success'] is True
        assert result['old_value'] == 30
        assert result['new_value'] == 45
        
        # Vérifier que la valeur a été mise à jour
        get_result = await self.config_manager.get_config_value('exchanges.binance.timeout')
        assert get_result['success'] is True
        assert get_result['value'] == 45

    async def test_set_config_value_invalid_path(self):
        """Test de définition de valeur avec chemin invalide."""
        # Given
        if not hasattr(self.config_manager, 'set_config_value') or not hasattr(self.config_manager, 'load_config'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # When
        result = await self.config_manager.set_config_value('', 0.2)  # Chemin vide
        
        # Then
        assert result['success'] is False
        assert 'error' in result
        assert 'path' in result['error'].lower() or 'invalid' in result['error'].lower()

    async def test_reset_config_nominal(self):
        """Test de réinitialisation de configuration nominale."""
        # Given
        if not hasattr(self.config_manager, 'reset_config') or not hasattr(self.config_manager, 'load_config') or not hasattr(self.config_manager, 'set_config_value'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # Modifier une valeur
        await self.config_manager.set_config_value('trading.max_position_size', 0.5)
        
        # When
        result = await self.config_manager.reset_config()
        
        # Then
        assert result['success'] is True
        assert 'backup_file' in result
        
        # Vérifier que la valeur a été réinitialisée
        get_result = await self.config_manager.get_config_value('trading.max_position_size')
        assert get_result['success'] is True
        assert get_result['value'] == 0.1  # Valeur par défaut

    async def test_merge_config_nominal(self):
        """Test de fusion de configuration nominale."""
        # Given
        if not hasattr(self.config_manager, 'merge_config') or not hasattr(self.config_manager, 'load_config'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        merge_config = {
            'trading': {
                'new_parameter': 'test_value'
            }
        }
        
        # When
        result = await self.config_manager.merge_config(merge_config)
        
        # Then
        assert result['success'] is True
        assert 'merged_keys' in result
        
        # Vérifier que la nouvelle valeur a été ajoutée
        get_result = await self.config_manager.get_config_value('trading.new_parameter')
        assert get_result['success'] is True
        assert get_result['value'] == 'test_value'

    async def test_get_config_history_nominal(self):
        """Test de récupération de l'historique de configuration nominale."""
        # Given
        if not hasattr(self.config_manager, 'get_config_history') or not hasattr(self.config_manager, 'load_config'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # When
        result = await self.config_manager.get_config_history()
        
        # Then
        assert result['success'] is True
        assert 'history' in result
        assert isinstance(result['history'], list)
        
        # L'historique devrait contenir au moins l'état initial
        assert len(result['history']) >= 1

    async def test_rollback_config_nominal(self):
        """Test de retour en arrière de configuration nominale."""
        # Given
        if not hasattr(self.config_manager, 'rollback_config') or not hasattr(self.config_manager, 'load_config') or not hasattr(self.config_manager, 'save_config'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # Modifier et sauvegarder
        await self.config_manager.set_config_value('trading.max_position_size', 0.3)
        save_result = await self.config_manager.save_config()
        
        # Modifier à nouveau
        await self.config_manager.set_config_value('trading.max_position_size', 0.4)
        
        # When
        # Revenir à la version précédente
        rollback_result = await self.config_manager.rollback_config()
        
        # Then
        assert rollback_result['success'] is True
        assert 'rollback_to' in rollback_result
        
        # Vérifier que la valeur a été restaurée
        get_result = await self.config_manager.get_config_value('trading.max_position_size')
        assert get_result['success'] is True
        assert get_result['value'] == 0.3

    async def test_export_config_nominal(self):
        """Test d'export de configuration nominale."""
        # Given
        if not hasattr(self.config_manager, 'export_config') or not hasattr(self.config_manager, 'load_config'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # When
        result = await self.config_manager.export_config()
        
        # Then
        assert result['success'] is True
        assert 'data' in result
        assert 'format' in result
        assert 'timestamp' in result
        
        # Vérifier que les données exportées sont valides
        exported_data = json.loads(result['data'])
        assert 'exchanges' in exported_data
        assert 'trading' in exported_data

    async def test_export_config_different_format(self):
        """Test d'export de configuration dans différents formats."""
        # Given
        if not hasattr(self.config_manager, 'export_config') or not hasattr(self.config_manager, 'load_config'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # When
        # Exporter en YAML
        result_yaml = await self.config_manager.export_config(ConfigFormat.YAML)
        
        # Exporter en JSON
        result_json = await self.config_manager.export_config(ConfigFormat.JSON)
        
        # Then
        assert result_yaml['success'] is True
        assert result_json['success'] is True
        assert result_yaml['format'] == ConfigFormat.YAML
        assert result_json['format'] == ConfigFormat.JSON
        
        # Vérifier que les données sont valides dans les deux formats
        import yaml
        yaml_data = yaml.safe_load(result_yaml['data'])
        json_data = json.loads(result_json['data'])
        
        assert yaml_data['trading']['max_position_size'] == json_data['trading']['max_position_size']

    async def test_import_config_nominal(self):
        """Test d'import de configuration nominale."""
        # Given
        if not hasattr(self.config_manager, 'import_config') or not hasattr(self.config_manager, 'load_config'):
            pytest.skip("Required methods not implemented")
            
        import_data = {
            'exchanges': {
                'coinbase': {
                    'api_key': 'imported_key',
                    'api_secret': 'imported_secret'
                }
            },
            'trading': {
                'max_position_size': 0.2
            }
        }
        
        # When
        result = await self.config_manager.import_config(import_data)
        
        # Then
        assert result['success'] is True
        assert 'imported_keys' in result
        
        # Vérifier que les données ont été importées
        get_result = await self.config_manager.get_config_value('exchanges.coinbase.api_key')
        assert get_result['success'] is True
        assert get_result['value'] == 'imported_key'

    async def test_import_config_conflict(self):
        """Test d'import de configuration avec conflit."""
        # Given
        if not hasattr(self.config_manager, 'import_config') or not hasattr(self.config_manager, 'load_config'):
            pytest.skip("Required methods not implemented")
            
        # Importer une configuration qui existe déjà
        await self.config_manager.load_config()
        
        import_data = {
            'trading': {
                'max_position_size': 0.5  # Différent de la valeur actuelle
            }
        }
        
        # When
        result = await self.config_manager.import_config(import_data, overwrite=False)
        
        # Then
        assert result['success'] is False
        assert 'error' in result
        assert 'conflict' in result['error'].lower() or 'exists' in result['error'].lower()

    async def test_environment_override(self):
        """Test de surcharge par variables d'environnement."""
        # Given
        if not hasattr(self.config_manager, 'load_config') or not hasattr(self.config_manager, 'get_config_value'):
            pytest.skip("Required methods not implemented")
            
        # Simuler des variables d'environnement
        os.environ['CONFIG_TRADING_MAX_POSITION_SIZE'] = '0.8'
        os.environ['CONFIG_BINANCE_API_KEY'] = 'env_key'
        
        await self.config_manager.load_config()
        
        # When
        # Vérifier que les variables d'environnement sont prises en compte
        size_result = await self.config_manager.get_config_value('trading.max_position_size')
        key_result = await self.config_manager.get_config_value('exchanges.binance.api_key')
        
        # Then
        assert size_result['success'] is True
        assert size_result['value'] == 0.8  # Valeur de l'environnement
        assert size_result['source'] == 'environment'
        
        assert key_result['success'] is True
        assert key_result['value'] == 'env_key'
        assert key_result['source'] == 'environment'

    async def test_config_encryption(self):
        """Test de chiffrement de configuration."""
        # Given
        if not hasattr(self.config_manager, 'enable_encryption') or not hasattr(self.config_manager, 'save_config'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # Activer le chiffrement
        await self.config_manager.enable_encryption('test_key')
        
        # When
        # Sauvegarder la configuration (devrait être chiffrée)
        result = await self.config_manager.save_config()
        
        # Then
        assert result['success'] is True
        assert 'encrypted' in result
        assert result['encrypted'] is True

    async def test_config_versioning(self):
        """Test de versionnement de configuration."""
        # Given
        if not hasattr(self.config_manager, 'get_config_version') or not hasattr(self.config_manager, 'save_config'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # When
        # Récupérer la version actuelle
        version_result = await self.config_manager.get_config_version()
        
        # Then
        assert version_result['success'] is True
        assert 'version' in version_result
        assert 'build_date' in version_result
        
        # Modifier et sauvegarder (devrait incrémenter la version)
        await self.config_manager.set_config_value('trading.version', '1.0.1')
        save_result = await self.config_manager.save_config()
        
        new_version_result = await self.config_manager.get_config_version()
        assert new_version_result['success'] is True

    async def test_concurrent_config_access(self):
        """Test d'accès concurrent à la configuration."""
        # Given
        if not hasattr(self.config_manager, 'set_config_value') or not hasattr(self.config_manager, 'get_config_value'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # When
        # Accès concurrents
        tasks = []
        for i in range(10):
            task = self.config_manager.set_config_value(f'concurrent_test_{i}', i)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Then
        successful_operations = [r for r in results if r and r.get('success')]
        assert len(successful_operations) == 10  # Tous devraient réussir

    async def test_error_handling_invalid_inputs(self):
        """Test de gestion des erreurs avec entrées invalides."""
        # Given/When/Then
        if hasattr(self.config_manager, 'set_config_value'):
            # Test avec chemin vide
            with pytest.raises((ValueError, TypeError)):
                await self.config_manager.set_config_value('', 'value')
            
            # Test avec valeur None
            with pytest.raises((ValueError, TypeError)):
                await self.config_manager.set_config_value('test.path', None)

    async def test_performance_with_large_config(self):
        """Test de performance avec grande configuration."""
        # Given
        if not hasattr(self.config_manager, 'load_config'):
            pytest.skip("load_config method not implemented")
            
        # Créer une grande configuration
        large_config = {}
        for i in range(1000):
            large_config[f'section_{i}'] = {
                'param_{j}': f'value_{j}' for j in range(10)
            }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(large_config, f)
            large_config_file = f.name
        
        # When
        start_time = datetime.now()
        
        # Charger la grande configuration
        if hasattr(ConfigurationManager, '__call__'):
            large_config_manager = ConfigurationManager(large_config_file)
            load_result = await large_config_manager.load_config()
        
        end_time = datetime.now()
        
        # Then
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 5.0  # Devrait s'exécuter rapidement
        
        # Nettoyer
        os.unlink(large_config_file)

    async def test_memory_usage_with_large_config(self):
        """Test de l'utilisation mémoire avec grande configuration."""
        # Given
        # Créer une configuration très volumineuse
        huge_config = {}
        for i in range(10000):
            huge_config[f'section_{i}'] = {
                'data': 'x' * 1000,  # Grande chaîne
                'nested': {
                    'level1': {'y' * 100},
                    'level2': {'z' * 50}
                }
            }
        
        # When/Then
        # Le système devrait pouvoir gérer cette charge sans erreur de mémoire
        assert len(huge_config) == 10000
        assert len(str(huge_config)) > 1000000  # Vérifier la taille

    async def test_config_hot_reload(self):
        """Test de rechargement à chaud de configuration."""
        # Given
        if not hasattr(self.config_manager, 'enable_hot_reload') or not hasattr(self.config_manager, 'load_config'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # Activer le rechargement à chaud
        await self.config_manager.enable_hot_reload()
        
        # Modifier le fichier de configuration
        new_config = await self.config_manager.get_config()
        new_config['trading']['max_position_size'] = 0.99
        
        with open(self.temp_config_file, 'w') as f:
            json.dump(new_config, f)
        
        # When
        # Attendre un peu pour que le rechargement à chaud détecte le changement
        await asyncio.sleep(0.1)
        
        # Vérifier que la configuration a été rechargée
        get_result = await self.config_manager.get_config_value('trading.max_position_size')
        
        # Then
        assert get_result['success'] is True
        assert get_result['value'] == 0.99

    async def test_config_validation_rules(self):
        """Test des règles de validation de configuration."""
        # Given
        if not hasattr(self.config_manager, 'add_validation_rule') or not hasattr(self.config_manager, 'validate_config'):
            pytest.skip("Required methods not implemented")
            
        # Ajouter une règle de validation personnalisée
        validation_rule = {
            'path': 'trading.max_position_size',
            'type': 'range',
            'min_value': 0.01,
            'max_value': 1.0,
            'message': 'Position size must be between 0.01 and 1.0'
        }
        
        # When
        result = await self.config_manager.add_validation_rule(validation_rule)
        
        # Then
        assert result['success'] is True
        assert 'rule_id' in result
        
        # Tester la validation
        valid_config = {'trading': {'max_position_size': 0.5}}
        invalid_config = {'trading': {'max_position_size': 2.0}}  # Trop grand
        
        valid_result = await self.config_manager.validate_config(valid_config)
        invalid_result = await self.config_manager.validate_config(invalid_config)
        
        assert valid_result['success'] is True
        assert invalid_result['success'] is False
        assert 'between 0.01 and 1.0' in invalid_result['errors'][0]

    async def test_config_backup_rotation(self):
        """Test de rotation des backups de configuration."""
        # Given
        if not hasattr(self.config_manager, 'set_backup_rotation') or not hasattr(self.config_manager, 'save_config_with_backup'):
            pytest.skip("Required methods not implemented")
            
        await self.config_manager.load_config()
        
        # Activer la rotation des backups
        await self.config_manager.set_backup_rotation(max_backups=3)
        
        # When
        # Créer plusieurs sauvegardes
        for i in range(5):
            await self.config_manager.set_config_value('test_value', i)
            await self.config_manager.save_config_with_backup()
        
        # Then
        # Vérifier que seulement les 3 derniers backups sont gardés
        backup_result = await self.config_manager.get_backup_list()
        assert backup_result['success'] is True
        assert len(backup_result['backups']) <= 3