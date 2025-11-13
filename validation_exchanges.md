# Rapport de Qualité de Migration des Assertions

Date: 2025-11-12 21:51:43

## Statistiques de Qualité

- Fichiers validés: 6
- Fichiers avec problèmes: 6
- Total des problèmes: 71
- Score moyen de qualité: 30.0/100

## Répartition des Problèmes par Sévérité

- ❌ Erreurs: 6
- ⚠️  Avertissements: 64
- ℹ️  Informations: 1

## Détail par Fichier

### 🔴 test_order_router.py - Score: 0/100 (À améliorer)

❌ **Errors** (1):
- Ligne N/A: Import des assertions standardisées manquant
  💡 Suggestion: Ajouter: from tests.utils import (assert_success_response, assert_error_response, ...)

⚠️ **Warnings** (11):
- Ligne 294: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 372: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 413: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 460: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 517: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 565: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 635: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 323: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 347: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 388: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 431: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)

### 🔴 test_trading_receiver.py - Score: 0/100 (À améliorer)

❌ **Errors** (1):
- Ligne N/A: Import des assertions standardisées manquant
  💡 Suggestion: Ajouter: from tests.utils import (assert_success_response, assert_error_response, ...)

⚠️ **Warnings** (14):
- Ligne 345: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 388: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 411: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 434: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 477: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 505: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 546: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 587: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 636: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 685: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 371: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 460: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 526: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 566: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)

### 🔴 test_unified_trading_standardizer.py - Score: 0/100 (À améliorer)

❌ **Errors** (1):
- Ligne N/A: Import des assertions standardisées manquant
  💡 Suggestion: Ajouter: from tests.utils import (assert_success_response, assert_error_response, ...)

⚠️ **Warnings** (17):
- Ligne 91: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 202: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 285: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 319: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 361: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 395: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 434: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 459: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 489: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 520: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 549: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 602: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 846: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 175: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 413: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 101: Comparaisons de prix sans tolérance
  Code: `assert std_order['price'] == 2000.0`
  💡 Suggestion: Remplacer par: assert_price_equals(actual, expected)
- Ligne 295: Comparaisons de prix sans tolérance
  Code: `assert std_trade['price'] == 2000.0`
  💡 Suggestion: Remplacer par: assert_price_equals(actual, expected)

ℹ️ **Infos** (1):
- Ligne 435: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(result['converted_amount'] - expected) < 0.0001`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)

### 🔴 test_config_manager.py - Score: 0/100 (À améliorer)

❌ **Errors** (1):
- Ligne N/A: Import des assertions standardisées manquant
  💡 Suggestion: Ajouter: from tests.utils import (assert_success_response, assert_error_response, ...)

⚠️ **Warnings** (22):
- Ligne 126: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 165: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 217: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 240: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 269: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 339: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 358: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 391: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 415: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 455: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 481: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 501: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 548: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 608: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 681: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 843: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 193: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 294: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 318: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 374: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 436: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 635: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)

### 🟢 test_exchange_dispatcher_refactored.py - Score: 90/100 (Excellent)

❌ **Errors** (1):
- Ligne N/A: Import des assertions standardisées manquant
  💡 Suggestion: Ajouter: from tests.utils import (assert_success_response, assert_error_response, ...)

### 🟢 test_exchange_dispatcher.py - Score: 90/100 (Excellent)

❌ **Errors** (1):
- Ligne N/A: Import des assertions standardisées manquant
  💡 Suggestion: Ajouter: from tests.utils import (assert_success_response, assert_error_response, ...)

## Recommandations d'Amélioration

1. **Corriger les erreurs critiques** avant de merger
2. **Compléter la migration** des assertions manuelles restantes
3. **Optimiser les comparaisons** numériques avec tolérances
4. **Valider les tests** après correction
5. **Documenter les patterns** spécifiques au projet
