# Guide des Patterns de Mock pour les Tests ARES

## 📋 Objectif

Ce guide documente les patterns et meilleures pratiques pour l'utilisation des mocks dans les tests du projet ARES, afin d'assurer la cohérence et d'éviter les régressions futures.

## 🏗️ Architecture des Mocks

### Structure des Fichiers

```
tests/utils/
├── assertions.py          # Assertions standardisées
├── mock_fixtures.py     # Mocks préconfigurés et utilitaires
└── __init__.py          # Export centralisé
```

### Classes de Mock Principales

1. **MockExchangeStatus** - Remplace l'enum ExchangeStatus
2. **MockOrderStatus** - Remplace l'enum OrderStatus
3. **MockOrderType** - Remplace l'enum OrderType
4. **MockOrderSide** - Remplace l'enum OrderSide
5. **MockSimulatorConfig** - Remplace la classe SimulatorConfig
6. **MockSlippageModel** - Remplace l'enum SlippageModel
7. **MockPaperTradingSimulator** - Remplace PaperTradingSimulator
8. **MockExchangeDispatcher** - Remplace ExchangeDispatcher
9. **MockOrderManager** - Remplace OrderManager

## 🔧 Patterns d'Importation

### Pattern 1: Import avec Fallback Sécurisé

```python
from tests.utils.mock_fixtures import (
    MockPaperTradingSimulator,
    MockSimulatorConfig,
    DependencyManager
)

# Import du module à tester avec fallback vers le mock
PaperTradingSimulator = DependencyManager.safe_import(
    'src.simulator.paper_trading_simulator.PaperTradingSimulator',
    fallback_class=MockPaperTradingSimulator
)
```

### Pattern 2: Configuration Centralisée

```python
def setup_method(self):
    """Setup pour chaque test."""
    # Utiliser le mock préconfiguré ou la vraie classe si disponible
    if hasattr(PaperTradingSimulator, '__call__') and PaperTradingSimulator is not Mock:
        self.simulator = PaperTradingSimulator(config)
    else:
        # Utiliser le mock préconfiguré
        self.simulator = MockPaperTradingSimulator(config)
```

## 🎯 Patterns de Configuration

### Pattern 1: Mock avec Side Effects Préconfigurés

```python
class MockPaperTradingSimulator:
    def __init__(self, config=None, exchange="binance", initial_balance=10000.0):
        self.config = config or MockSimulatorConfig()
        self.exchange = exchange
        self.initial_balance = initial_balance
        
        # Configuration des méthodes asynchrones
        self.simulate_order = AsyncMock(side_effect=self._simulate_order_side_effect)
        self.get_positions = Mock(return_value=[])
        self.get_performance_metrics = Mock(return_value=self._get_default_metrics())
    
    def _simulate_order_side_effect(self, symbol, side, order_type, quantity, price, order_book):
        """Side effect pour simulate_order."""
        # Logique de test complète ici
        if quantity > 1000:
            return {'status': MockOrderStatus.REJECTED, 'reason': 'Insufficient balance'}
        return {'status': MockOrderStatus.FILLED, 'order_id': f'order_{uuid.uuid4().hex[:8]}'}
```

### Pattern 2: Validation des Attributs

```python
# Toujours vérifier les attributs avant de les utiliser
if hasattr(self.simulator, '_running'):
    if hasattr(self.simulator._running, 'return_value'):
        # AsyncMock
        assert self.simulator._running.return_value is True
    else:
        # Attribut normal
        assert self.simulator._running is True
```

## 📝 Patterns d'Assertions

### Pattern 1: Utilisation des Constantes de Mock

```python
# ❌ À éviter
assert_order_status(result['status'], "FILLED", "L'ordre doit être rempli")

# ✅ Recommandé
assert_order_status(result['status'], MockOrderStatus.FILLED, "L'ordre doit être rempli")
```

### Pattern 2: Assertions Structurées

```python
# Utiliser les assertions standardisées pour la cohérence
from tests.utils import assert_dict_structure, assert_list_structure

# Vérifier la structure des réponses
assert_dict_structure(
    result,
    required_keys=['order_id', 'symbol', 'status', 'timestamp'],
    message="La réponse doit contenir les clés requises"
)

assert_list_structure(
    orders,
    min_length=1,
    item_type=dict,
    message="La liste d'ordres doit contenir au moins un élément"
)
```

## 🔄 Patterns de Gestion des Erreurs

### Pattern 1: Gestion des Imports Manquants

```python
# Utiliser DependencyManager pour les imports
DependencyManager.safe_import(
    'module.path.ClassName',
    fallback_class=MockClassName
)
```

### Pattern 2: Messages d'Erreur Descriptifs

```python
# Toujours inclure des messages d'erreur clairs
assert_order_status(
    result['status'],
    MockOrderStatus.REJECTED,
    message="L'ordre avec quantité invalide doit être rejeté"
)
```

## 🧪 Patterns de Test

### Pattern 1: Tests d'Intégration avec Mocks

```python
@pytest.mark.asyncio
async def test_order_creation_with_mock(self, mock_order_data):
    """Test de création d'ordre avec mock."""
    # Given
    order_data = mock_order_data
    
    # When
    result = await self.order_manager.create_order(
        order_data['symbol'],
        order_data['side'],
        order_data['order_type'],
        order_data['quantity'],
        order_data['price']
    )
    
    # Then
    assert_success_response(result, "La création d'ordre devrait réussir")
    assert_order_status(result['order']['status'], MockOrderStatus.OPEN)
```

### Pattern 2: Tests de Performance

```python
async def test_performance_with_many_orders(self):
    """Test de performance avec beaucoup d'ordres."""
    # Given
    start_time = datetime.now()
    
    # When
    tasks = [
        self.order_manager.create_order(f'SYMBOL{i}', 'buy', 'market', 0.1, 2000.0)
        for i in range(100)
    ]
    await asyncio.gather(*tasks)
    
    # Then
    execution_time = (datetime.now() - start_time).total_seconds()
    assert_execution_time(execution_time, 10.0, "L'exécution doit prendre moins de 10 secondes")
```

## 🚨 Pièges à Éviter

### 1. Hardcoding des Valeurs

```python
# ❌ À éviter
assert result['status'] == 'FILLED'

# ✅ Recommandé
assert_order_status(result['status'], MockOrderStatus.FILLED)
```

### 2. Mocks Incomplets

```python
# ❌ À éviter
mock_obj = AsyncMock()  # Sans configuration

# ✅ Recommandé
mock_obj = MockPaperTradingSimulator(config)  # Préconfiguré
```

### 3. Tests Fragiles

```python
# ❌ À éviter
# Dépendre de l'ordre exact des méthodes

# ✅ Recommandé
# Vérifier les attributs existants avant de les utiliser
if hasattr(self.order_manager, 'create_order'):
    await self.order_manager.create_order(...)
```

## 📊 Métriques de Qualité

### Indicateurs à Suivre

1. **Couverture de Mock**: Pourcentage de classes mockées
2. **Consistance des Assertions**: Utilisation des assertions standardisées
3. **Réutilisabilité**: Nombre de mocks réutilisés entre tests
4. **Maintenance**: Facilité de mise à jour des mocks

### Objectifs Cibles

- 90% des tests utilisent les mocks standardisés
- 100% des assertions utilisent les fonctions standardisées
- 0 régression due aux changements de mock
- Temps de maintenance des mocks < 1 heure par mois

## 🔧 Outils d'Aide

### DependencyManager

```python
# Import sécurisé avec fallback
DependencyManager.safe_import(module_path, fallback_class, fallback_value)

# Création de mocks pour classes manquantes
DependencyManager.create_mock_for_missing_class(class_name, base_class)

# Patch automatique de modules
DependencyManager.patch_missing_module(module_path, mock_class)
```

### MockHelpers

```python
# Configuration d'AsyncMock avec side effects
MockHelpers.configure_async_mock_with_side_effect(mock_obj, side_effect_func)

# Configuration d'attributs de Mock
MockHelpers.configure_mock_attributes(mock_obj, attributes_dict)

# Création de mocks avec méthodes préconfigurées
MockHelpers.create_mock_with_methods(methods_dict)
```

## 📚 Références et Ressources

### Documentation Complémentaire

1. **Documentation pytest**: https://docs.pytest.org/
2. **Mock unittest**: https://docs.python.org/3/library/unittest.mock.html
3. **AsyncIO**: https://docs.python.org/3/library/asyncio.html

### Exemples dans le Projet

- `tests/unit/test_simulator/test_paper_trading_simulator.py`
- `tests/unit/test_exchanges/test_exchange_dispatcher.py`
- `tests/unit/test_trading/test_order_manager.py`

## 🔄 Processus de Mise à Jour

### Quand Modifier un Mock

1. **Identifier l'impact**: Quels tests sont affectés ?
2. **Mettre à jour le mock**: Modifier la classe dans `mock_fixtures.py`
3. **Adapter les tests**: Utiliser les nouvelles constantes/méthodes
4. **Vérifier la régression**: Exécuter les tests concernés
5. **Documenter les changements**: Mettre à jour ce guide

### Quand Ajouter un Mock

1. **Analyser le besoin**: Quelle fonctionnalité doit être mockée ?
2. **Créer la classe**: Implémenter dans `mock_fixtures.py`
3. **Ajouter les exports**: Mettre à jour `__init__.py`
4. **Créer les tests**: Écrire des tests utilisant le nouveau mock
5. **Documenter**: Ajouter les patterns à ce guide

---

**Dernière mise à jour**: 12 novembre 2025  
**Auteur**: Équipe de développement ARES  
**Version**: 1.0