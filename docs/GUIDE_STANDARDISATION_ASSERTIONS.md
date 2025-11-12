# Guide d'Utilisation des Assertions Standardisées - Projet ARES

## Introduction

Ce guide présente les utilitaires d'assertion standardisés créés pour améliorer la fiabilité et la cohérence des tests unitaires dans le projet ARES. L'utilisation de ces helpers permet d'éviter les erreurs courantes liées aux assertions numériques, structures de données et formats.

## Installation

Les utilitaires d'assertion sont disponibles dans le module `tests.utils.assertions` :

```python
from tests.utils.assertions import (
    assert_success_response,
    assert_error_response,
    assert_float_equals,
    assert_price_equals,
    assert_dict_structure,
    assert_execution_time,
    assert_order_status,
    assert_performance_metrics,
    assert_timestamp_format,
    assert_list_structure,
    assert_exchange_status,
    assert_signal_status,
    assert_dataframe_structure
)
```

## Principes de Base

### 1. Tolérances Appropriées

Les comparaisons numériques doivent toujours utiliser des tolérances appropriées :

```python
# ❌ Mauvais - comparaison exacte de flottants
assert result['price'] == 2000.0

# ✅ Bon - utilisation de tolérance
assert_float_equals(result['price'], 2000.0, tolerance=1e-4)
assert_price_equals(result['price'], 2000.0)  # Tolérance automatique pour les prix
```

### 2. Messages d'Erreur Clairs

Toutes les assertions doivent inclure des messages d'erreur descriptifs :

```python
# ❌ Mauvais - message générique
assert result['success'] is True

# ✅ Bon - message descriptif
assert_success_response(result, "Le dispatch devrait réussir")
```

### 3. Validation de Structure

Validez toujours la structure des données avant de vérifier les valeurs :

```python
# ❌ Mauvais - accès direct sans validation
assert result['data']['price'] == 2000.0

# ✅ Bon - validation de structure d'abord
assert_dict_structure(result, ['success', 'data'])
assert_float_equals(result['data']['price'], 2000.0)
```

## Catégories d'Assertions

### 1. Assertions de Réponse API

#### Réponses Succès
```python
assert_success_response(response, "L'opération devrait réussir")
```

#### Réponses d'Erreur
```python
assert_error_response(
    response, 
    expected_error_substring="not found",
    message="L'opération devrait échouer avec une erreur claire"
)
```

### 2. Assertions Numériques

#### Comparaisons Flottantes
```python
# Tolérance par défaut (1e-6)
assert_float_equals(actual, expected)

# Tolérance personnalisée
assert_float_equals(actual, expected, tolerance=1e-3)

# Pourcentage
assert_percentage_equals(actual_percentage, expected_percentage)

# Prix (tolérance relative automatique)
assert_price_equals(actual_price, expected_price)
```

### 3. Assertions de Structure

#### Dictionnaires
```python
# Clés requises uniquement
assert_dict_structure(data, ['success', 'order_id', 'status'])

# Clés requises et optionnelles
assert_dict_structure(
    data, 
    required_keys=['success', 'order_id'],
    optional_keys=['timestamp', 'metadata']
)
```

#### Listes
```python
# Validation de longueur
assert_list_structure(items, min_length=1, max_length=10)

# Validation de type d'éléments
assert_list_structure(items, item_type=dict)
```

#### DataFrames pandas
```python
assert_dataframe_structure(
    df, 
    expected_columns=['timestamp', 'price', 'volume'],
    min_rows=100
)
```

### 4. Assertions de Format

#### Timestamps
```python
# Format ISO
assert_timestamp_format(timestamp, format_type="iso")

# Format UNIX
assert_timestamp_format(timestamp, format_type="unix")

# Format datetime
assert_timestamp_format(timestamp, format_type="datetime")
```

### 5. Assertions de Statut

#### Statuts d'Ordre
```python
assert_order_status(actual_status, 'FILLED')
assert_order_status(actual_status, OrderStatus.FILLED)  # Avec énumération
```

#### Statuts d'Exchange
```python
assert_exchange_status(actual_status, 'ACTIVE')
assert_exchange_status(actual_status, ExchangeStatus.ACTIVE)  # Avec énumération
```

#### Statuts de Signal
```python
assert_signal_status(actual_status, 'PROCESSED')
assert_signal_status(actual_status, SignalStatus.PROCESSED)  # Avec énumération
```

### 6. Assertions de Performance

#### Métriques de Performance
```python
assert_performance_metrics(metrics)
assert_performance_metrics(metrics, required_metrics=['total_return', 'sharpe_ratio'])
```

#### Temps d'Exécution
```python
assert_execution_time(execution_time, max_time=5.0)
```

## Patterns Recommandés

### 1. Tests de Méthodes Asynchrones

```python
@pytest.mark.asyncio
async def test_async_operation(self):
    # Given
    if not hasattr(self.service, 'async_method'):
        pytest.skip("async_method not implemented")
    
    # When
    result = await self.service.async_method()
    
    # Then - Utilisation des assertions standardisées
    assert_success_response(result)
    assert_dict_structure(result, ['success', 'data', 'timestamp'])
    assert_timestamp_format(result['timestamp'], format_type="datetime")
```

### 2. Tests de Cas d'Erreur

```python
@pytest.mark.asyncio
async def test_error_case(self):
    # Given
    invalid_input = {"invalid": "data"}
    
    # When/Then
    with pytest.raises((ValueError, TypeError)):
        await self.service.process(invalid_input)
    
    # Pour les retours d'erreur API
    result = await self.service.process_invalid_input()
    assert_error_response(result, expected_error_substring="invalid")
```

### 3. Tests de Performance

```python
@pytest.mark.asyncio
async def test_performance_with_large_dataset(self):
    # Given
    large_dataset = self._create_large_dataset()
    
    # When
    start_time = datetime.now()
    result = await self.service.process_large_dataset(large_dataset)
    end_time = datetime.now()
    
    # Then
    execution_time = (end_time - start_time).total_seconds()
    assert_execution_time(execution_time, max_time=10.0)
    assert_success_response(result)
```

### 4. Tests Concurrents

```python
@pytest.mark.asyncio
async def test_concurrent_operations(self):
    # Given
    tasks = [self.service.process(item) for item in items]
    
    # When
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Then
    successful_operations = [r for r in results if r and r.get('success')]
    assert_list_structure(
        successful_operations,
        min_length=len(items),
        message="Toutes les opérations concurrentes devraient réussir"
    )
```

## Migration depuis les Assertions Traditionnelles

### Avant (Inconsistant)
```python
# Valeurs magiques et tolérances hardcodées
assert result['success'] is True
assert len(result['orders']) == 2
assert result['price'] == 2000.0
assert execution_time < 5.0
assert 'order_id' in result
assert result['status'] == 'FILLED'
```

### Après (Standardisé)
```python
# Assertions claires avec tolérances appropriées
assert_success_response(result, "L'opération devrait réussir")
assert_dict_structure(result, ['success', 'orders'])
assert_list_structure(result['orders'], min_length=2, max_length=2)
assert_price_equals(result['price'], 2000.0)
assert_execution_time(execution_time, max_time=5.0)
assert_order_status(result['status'], 'FILLED')
```

## Bonnes Pratiques

### 1. Messages d'Erreur Descriptifs
- Toujours inclure un message qui explique ce qui est testé
- Utiliser des messages en français pour la cohérence avec le projet
- Inclure le contexte de l'erreur quand possible

### 2. Tolérances Appropriées
- Utiliser `assert_price_equals()` pour les prix financiers
- Utiliser `assert_float_equals()` avec une tolérance explicite pour les valeurs générales
- Éviter les comparaisons exactes de nombres flottants

### 3. Validation de Structure
- Valider toujours la structure avant les valeurs
- Utiliser `assert_dict_structure()` pour les réponses API
- Utiliser `assert_list_structure()` pour les collections

### 4. Tests Asynchrones
- Utiliser `@pytest.mark.asyncio` pour les tests asynchrones
- Gérer les cas où les méthodes ne sont pas implémentées avec `pytest.skip()`

### 5. Tests d'Erreur
- Tester les cas d'erreur attendus
- Utiliser `assert_error_response()` pour les erreurs API
- Utiliser `pytest.raises()` pour les exceptions

## Exemples Complets

### Test d'Exchange Dispatcher

```python
@pytest.mark.asyncio
async def test_dispatch_order_success(self):
    """Test de dispatch d'ordre réussi avec assertions standardisées."""
    # Given
    order = {
        'symbol': 'ETHUSDT',
        'side': 'buy',
        'order_type': 'market',
        'quantity': 0.1,
        'price': 2000.0
    }
    
    # When
    result = await self.dispatcher.dispatch_to_exchange(
        'binance', order['symbol'], order['side'], 
        order['order_type'], order['quantity'], order['price']
    )
    
    # Then - Assertions standardisées
    assert_success_response(result, "Le dispatch d'ordre devrait réussir")
    
    # Structure de la réponse
    assert_dict_structure(
        result,
        required_keys=['success', 'order_id', 'exchange', 'status', 'timestamp'],
        message="La réponse doit contenir toutes les clés requises"
    )
    
    # Valeurs spécifiques
    assert result['exchange'] == 'binance', "L'exchange devrait être 'binance'"
    assert result['symbol'] == 'ETHUSDT', "Le symbole devrait être 'ETHUSDT'"
    assert_order_status(result['status'], 'SUBMITTED', "Le statut devrait être 'SUBMITTED'")
    
    # Format du timestamp
    assert_timestamp_format(
        result['timestamp'],
        format_type="datetime",
        message="Le timestamp devrait être au format datetime"
    )
```

### Test de Performance

```python
@pytest.mark.asyncio
async def test_performance_metrics_calculation(self):
    """Test de calcul des métriques de performance avec assertions standardisées."""
    # Given
    trades = self._create_test_trades(100)
    
    # When
    start_time = datetime.now()
    metrics = await self.analyzer.calculate_performance_metrics(trades)
    execution_time = (end_time - start_time).total_seconds()
    
    # Then - Assertions standardisées
    assert_execution_time(
        execution_time, 
        max_time=5.0,
        message="Le calcul des métriques devrait prendre moins de 5 secondes"
    )
    
    # Structure des métriques
    assert_performance_metrics(
        metrics,
        required_metrics=['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'],
        message="Les métriques doivent contenir les indicateurs clés"
    )
    
    # Valeurs numériques avec tolérances
    assert_float_equals(
        metrics['total_return'], 
        0.15, 
        tolerance=0.01,
        message="Le retour total devrait être de 15% avec une tolérance de 1%"
    )
    
    assert_float_equals(
        metrics['sharpe_ratio'], 
        1.5, 
        tolerance=0.1,
        message="Le ratio de Sharpe devrait être de 1.5 avec une tolérance de 0.1"
    )
```

## Prochaines Étapes et Feuille de Route

### Phase 1 - Adoption Immédiate (À faire maintenant)
- [ ] **Utiliser les assertions standardisées** pour tout nouveau test créé
- [ ] **Partager ce guide** avec les membres de l'équipe
- [ ] **Ajouter les patterns** dans les checklists de code review

### Phase 2 - Migration Prioritaire (Semaines prochaines)
- [ ] **Identifier les tests critiques** les plus sujets aux erreurs d'assertion
- [ ] **Refactoriser 5-10 tests** prioritaires en utilisant les nouveaux patterns
- [ ] **Mesurer l'impact** sur la réduction des erreurs de tests

### Phase 3 - Déploiement Complet (Mois prochains)
- [ ] **Migrer tous les tests existants** vers les assertions standardisées
- [ ] **Créer des assertions spécifiques** pour les cas d'usage du projet ARES
- [ ] **Intégrer la validation** dans les pipelines CI/CD

### Phase 4 - Amélioration Continue
- [ ] **Collecter les retours** d'expérience des développeurs
- [ ] **Étendre la bibliothèque** avec de nouvelles assertions
- [ ] **Maintenir la documentation** à jour avec les évolutions

## Checklist de Migration Rapide

### Pour les Nouveaux Tests ✅
- [ ] Importer les assertions depuis `tests.utils`
- [ ] Utiliser `assert_success_response()` pour les réponses API réussies
- [ ] Utiliser `assert_float_equals()` avec tolérance pour les comparaisons numériques
- [ ] Utiliser `assert_dict_structure()` pour valider les structures de données
- [ ] Ajouter des messages d'erreur descriptifs en français

### Pour les Tests Existants 🔄
- [ ] Remplacer `assert x == y` par `assert_float_equals(x, y, tolerance)`
- [ ] Remplacer `assert 'key' in dict` par `assert_dict_structure(dict, ['key'])`
- [ ] Remplacer les validations de statut par les fonctions spécialisées
- [ ] Ajouter des messages d'erreur clairs

## Conclusion

L'utilisation de ces assertions standardisées permet de :
- **Améliorer la fiabilité** des tests en évitant les erreurs de précision numérique
- **Augmenter la lisibilité** avec des messages d'erreur clairs
- **Garantir la cohérence** des validations de structure
- **Faciliter la maintenance** avec des patterns réutilisables

Pour toute question ou suggestion d'amélioration, veuillez contacter l'équipe de développement ou créer une issue dans le suivi du projet.

---

**Date de mise à jour** : 12 novembre 2025
**Version du guide** : 1.1
**Prochaines mises à jour** : Basées sur les retours d'expérience des développeurs