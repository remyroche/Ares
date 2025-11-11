# Diagnostic de l'erreur "Found array with 0 sample(s)" lors du training analyst base

## Résumé du problème

Lors de l'exécution de `python3 src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT --execution-mode light`, l'erreur "Found array with 0 sample(s) (shape=(0, 60))" ou "Found array with 0 sample(s) (shape=(0, 9))" apparaît systématiquement lors de l'évaluation des objectifs. Le dataset contient 75 échantillons au total, ce qui est déjà insuffisant, mais l'erreur semble provenir d'un problème plus profond dans le pipeline de validation croisée.

## Analyse des logs

### Observations clés
1. **Configuration du mode light** : `cv_folds=2` (confirmé dans les logs)
2. **Nombre d'échantillons** : `Training samples: 75` (constant dans tous les logs)
3. **Features** : Soit 9 features, soit 60 features selon les exécutions
4. **Erreur systématique** : Tous les essais se terminent avec `Best score: 0.000000`
5. **Localisation de l'erreur** : L'erreur se produit dans `System.HPOConfig` lors de la prédiction du modèle

### Patterns observés
- L'erreur apparaît lors de `Model predict failed` ou `Objective evaluation failed`
- Deux formes d'erreur : `shape=(0, 60)` et `shape=(0, 9)` selon le nombre de features
- L'erreur se produit à la fois dans les phases de grille grossière et fine
- Le système détecte le problème : `🔍 LIGHT MODE DIAGNOSTIC: Very small CV folds (2) may cause score variance issues`

## Causes possibles identifiées

### 1. Validation croisée avec 2 folds sur dataset insuffisant (Cause principale - Probabilité très élevée)
- **Mécanisme** : Le mode light utilise `cv_folds=2` mais avec 75 échantillons, chaque fold ne contient qu'environ 37-38 échantillons
- **Impact** : Une répartition légèrement déséquilibrée peut laisser un fold avec très peu ou zéro échantillons
- **Preuve** : Les logs montrent systématiquement "Found array with 0 sample(s)" lors de l'évaluation des objectifs et "Training samples: 75" avec "cv_folds=2"

### 2. Pipeline de préparation des données trop restrictif en mode light (Cause secondaire - Probabilité élevée)
- **Mécanisme** : Le mode light limite à 20 jours de données vs 1460 en mode full
- **Impact** : Réduction drastique du nombre d'échantillons disponibles
- **Preuve** : Le dataset final ne contient que 75 échantillons

### 3. Problème d'extraction des features et labels de régime (Cause potentielle - Probabilité moyenne)
- **Mécanisme** : Les logs montrent des erreurs lors de la prédiction avec des tableaux de forme (0, 60) ou (0, 9)
- **Impact** : L'extraction des labels de régime peut échouer ou retourner des données vides
- **Preuve** : L'erreur se produit lors de `Model predict failed` dans `System.HPOConfig`

### 4. Répartition déséquilibrée des données entre train/validation (Cause potentielle - Probabilité moyenne)
- **Mécanisme** : Avec seulement 75 échantillons et 2 folds, la répartition peut être inégale
- **Impact** : Un fold peut se retrouver avec 0 échantillon après la séparation
- **Preuve** : Forme de l'erreur `shape=(0, 60)` indique un tableau vide

### 5. Problème dans la fonction objective d'évaluation (Cause potentielle - Probabilité moyenne)
- **Mécanisme** : La fonction objective dans `hpo_config.py` peut échouer lors de l'évaluation
- **Impact** : L'évaluation retourne un score nul par défaut
- **Preuve** : Les logs montrent "Objective evaluation failed" avec la même erreur

## Relation avec les scores de 0.000000

Le mécanisme est clair :
1. Lorsque l'évaluation échoue avec "Found array with 0 sample(s)", le système attribue un score nul par défaut
2. Tous les essais échouent, donc tous les scores sont 0.000000
3. L'optimiseur ne peut pas distinguer les configurations et retourne le score par défaut

## Diagnostic hiérarchisé

### Cause principale (Probabilité très élevée) : Validation croisée avec 2 folds sur dataset insuffisant
- **Mécanisme** : Le mode light utilise `cv_folds=2` mais avec 75 échantillons, chaque fold n'en contient qu'environ 37-38
- **Impact** : Une répartition légèrement déséquilibrée peut laisser un fold avec 0 échantillon
- **Preuve** : Les logs montrent systématiquement "Found array with 0 sample(s)" lors de l'évaluation des objectifs et "Training samples: 75" avec "cv_folds=2"

### Cause secondaire (Probabilité élevée) : Pipeline de préparation des données trop restrictif
- **Mécanisme** : Le mode light limite à 20 jours de données vs 1460 en mode full
- **Impact** : Réduction drastique du nombre d'échantillons disponibles
- **Preuve** : Le dataset final ne contient que 75 échantillons

## Logs recommandés pour valider les hypothèses

### 1. Dans le `HierarchicalParameterOptimizer` (lignes 1372-1397)
Ajouter des logs pour suivre la taille de chaque fold de validation croisée :

```python
# Avant la validation croisée
logger.info(f"🔍 CV DEBUG: Total samples={len(X_train)}, cv_folds={self.cv_folds}")
logger.info(f"🔍 CV DEBUG: Expected samples per fold={len(X_train) // self.cv_folds}")

# Dans la boucle de validation croisée
for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    logger.info(f"🔍 CV DEBUG: Fold {fold_idx+1}/{self.cv_folds}: train_size={len(train_idx)}, val_size={len(val_idx)}")
    if len(val_idx) == 0:
        logger.error(f"🚨 CV CRITICAL: Fold {fold_idx+1} has 0 validation samples!")
```

### 2. Dans le pipeline de validation croisée (`unified_cv.py`, lignes 85-95)
Ajouter des logs pour détecter les folds vides :

```python
# Après la création des folds
if hasattr(cv, 'split') and len(list(cv.split(X, y))) > 0:
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        if len(val_idx) == 0:
            LOGGER.error(f"🚨 CV CRITICAL: Fold {i+1} has 0 validation samples! X.shape={X.shape}")
            LOGGER.error(f"🚨 CV CRITICAL: Train indices: {train_idx}")
            LOGGER.error(f"🚨 CV CRITICAL: Val indices: {val_idx}")
```

### 3. Dans le pipeline de préparation des données (`hpo_config.py`, lignes 348-361)
Ajouter des logs pour suivre l'impact du filtrage :

```python
# Après la génération des features
logger.info(f"🔍 DATA DEBUG: Raw features shape: {features.shape}")

# Après l'extraction des features économiques
logger.info(f"🔍 DATA DEBUG: Economic features shape: {features_economic.shape}")
logger.info(f"🔍 DATA DEBUG: Economic features index range: {features_economic.index.min()} to {features_economic.index.max()}")

# Avant la prédiction
logger.info(f"🔍 PREDICT DEBUG: About to predict on features_economic.shape={features_economic.values.shape}")
```

### 4. Dans le processus d'extraction des labels de régime (`hpo_config.py`, lignes 518-520)
Ajouter des logs pour identifier les données vides :

```python
# Après la prédiction des labels de régime
logger.info(f"🔍 REGIME DEBUG: Regime labels shape: {regime_labels.shape}")
logger.info(f"🔍 REGIME DEBUG: Unique regimes: {np.unique(regime_labels)}")
logger.info(f"🔍 REGIME DEBUG: Regime distribution: {dict(zip(*np.unique(regime_labels, return_counts=True)))}")

# Avant l'évaluation de la qualité
if len(regime_labels) == 0:
    logger.error(f"🚨 REGIME CRITICAL: Empty regime labels detected!")
```

## Solutions recommandées

### Solution immédiate (temporaire)
1. **Augmenter le nombre minimum d'échantillons en mode light** : Passer de 20 jours à 60 jours
2. **Réduire le nombre de folds en mode light** : Utiliser `cv_folds=2` seulement si plus de 150 échantillons
3. **Ajouter une validation** : Vérifier que chaque fold contient au moins 10 échantillons

### Solution structurelle
1. **Ajustement dynamique des folds** : Adapter le nombre de folds selon la taille du dataset
2. **Validation de la répartition** : Vérifier que tous les folds contiennent des données avant l'évaluation
3. **Mode light amélioré** : Conserver plus de données ou utiliser une stratégie d'évaluation alternative

## Conclusion

L'erreur "Found array with 0 sample(s)" n'est pas seulement due au manque d'échantillons, mais principalement à une inadéquation entre la configuration de validation croisée (2 folds) et la taille du dataset (75 échantillons) en mode light. Le pipeline de préparation des données trop restrictif aggrave ce problème en réduisant davantage le nombre d'échantillons disponibles.

Les logs recommandés permettront de confirmer cette hypothèse et d'identifier précisément où se produit la rupture dans le pipeline.