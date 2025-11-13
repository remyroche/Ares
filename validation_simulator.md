# Rapport de Qualité de Migration des Assertions

Date: 2025-11-12 21:51:38

## Statistiques de Qualité

- Fichiers validés: 3
- Fichiers avec problèmes: 3
- Total des problèmes: 61
- Score moyen de qualité: 30.0/100

## Répartition des Problèmes par Sévérité

- ❌ Erreurs: 3
- ⚠️  Avertissements: 41
- ℹ️  Informations: 17

## Détail par Fichier

### 🔴 test_position_manager.py - Score: 0/100 (À améliorer)

❌ **Errors** (1):
- Ligne N/A: Import des assertions standardisées manquant
  💡 Suggestion: Ajouter: from tests.utils import (assert_success_response, assert_error_response, ...)

⚠️ **Warnings** (21):
- Ligne 91: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 122: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 240: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 276: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 297: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 316: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 345: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 369: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 410: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 438: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 460: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 513: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 540: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 561: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 705: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 145: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 164: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 202: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 222: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 259: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 391: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)

ℹ️ **Infos** (6):
- Ligne 376: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(result['unrealized_pnl'] - expected_pnl) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 419: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(result['pnl'] - expected_pnl) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 420: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(result['pnl_pct'] - expected_pct) < 0.0001`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 421: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(result['unrealized_pnl'] - expected_pnl) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 444: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(result['pnl'] - expected_pnl) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 522: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(position['entry_price'] - expected_entry_price) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)

### 🔴 test_fee_calculator.py - Score: 0/100 (À améliorer)

❌ **Errors** (1):
- Ligne N/A: Import des assertions standardisées manquant
  💡 Suggestion: Ajouter: from tests.utils import (assert_success_response, assert_error_response, ...)

⚠️ **Warnings** (20):
- Ligne 92: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 122: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 205: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 254: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 286: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 314: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 337: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 359: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 377: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 396: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 424: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 471: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 505: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 540: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 570: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 600: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 630: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 732: Assertions de succès manuelles
  Code: `assert result['success'] is True`
  💡 Suggestion: Remplacer par: assert_success_response(result)
- Ligne 182: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)
- Ligne 227: Assertions d'erreur manuelles
  Code: `assert result['success'] is False`
  💡 Suggestion: Remplacer par: assert_error_response(result)

ℹ️ **Infos** (11):
- Ligne 99: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(result['fee_amount'] - expected_amount) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 124: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(result['fee_amount'] - expected_amount) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 160: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(binance_result['fee_amount'] - expected_binance) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 161: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(okx_result['fee_amount'] - expected_okx) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 288: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(result['fee_amount'] - expected_fee) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 429: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(result['total_volume'] - expected_total) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 519: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(result['savings'] - expected_savings) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 550: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(fee_result['fee_amount'] - expected_fee) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 610: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(result['fee_amount'] - expected_fee) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 642: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(result['fee_amount'] - expected_fee) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)
- Ligne 773: Comparaisons de flottants sans tolérance explicite
  Code: `assert abs(fee_result['fee_amount'] - expected_fee) < 0.01`
  💡 Suggestion: Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)

### 🟢 test_paper_trading_simulator.py - Score: 90/100 (Excellent)

❌ **Errors** (1):
- Ligne N/A: Import des assertions standardisées manquant
  💡 Suggestion: Ajouter: from tests.utils import (assert_success_response, assert_error_response, ...)

## Recommandations d'Amélioration

1. **Corriger les erreurs critiques** avant de merger
2. **Compléter la migration** des assertions manuelles restantes
3. **Optimiser les comparaisons** numériques avec tolérances
4. **Valider les tests** après correction
5. **Documenter les patterns** spécifiques au projet
