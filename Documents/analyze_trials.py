import re
from typing import List, Dict, Tuple

def parse_trial_data(trial_text: str) -> List[Dict]:
    """Parse les données des essais à partir du texte fourni"""
    trials = []
    lines = trial_text.strip().split('\n')
    
    for line in lines:
        # Nettoyer la ligne en enlevant le tiret au début
        clean_line = line.lstrip('- ').strip()
        
        # Utiliser une expression régulière pour extraire les données
        match = re.match(
            r'Trial (\d+): Quality ([\d.]+) \| Regimes: (\d+) \| CV: ([\d.]+) \| EWMA=(\d+)\+(\d+) \| states=(\d+) \| κ=([\d.]+) \| min_cov=([\d.e-]+)',
            clean_line
        )
        
        if match:
            trial = {
                'trial_number': int(match.group(1)),
                'quality': float(match.group(2)),
                'regimes': int(match.group(3)),
                'cv': float(match.group(4)),
                'ewma_short': int(match.group(5)),
                'ewma_long': int(match.group(6)),
                'states': int(match.group(7)),
                'kappa': float(match.group(8)),
                'min_cov': float(match.group(9))
            }
            trials.append(trial)
        else:
            print(f"Échec du parsing pour: {clean_line}")
    
    return trials

def analyze_top_performers(trials: List[Dict], top_n: int = 5) -> List[Dict]:
    """Analyse et retourne les meilleurs performeurs"""
    # Trier par qualité (décroissant)
    sorted_by_quality = sorted(trials, key=lambda x: x['quality'], reverse=True)
    
    return sorted_by_quality[:top_n]

def analyze_by_criteria(trials: List[Dict]) -> Dict[str, List[Dict]]:
    """Analyse les performeurs selon différents critères"""
    results = {}
    
    # Meilleures qualités
    results['top_quality'] = sorted(trials, key=lambda x: x['quality'], reverse=True)[:5]
    
    # Plus faibles CV (Coefficient of Variation - plus stable)
    results['lowest_cv'] = sorted(trials, key=lambda x: x['cv'])[:5]
    
    # Meilleurs équilibres qualité/CV (qualité élevée avec CV faible)
    results['best_quality_cv_balance'] = sorted(
        trials, 
        key=lambda x: x['quality'] / (x['cv'] + 0.01),  # +0.01 pour éviter division par zéro
        reverse=True
    )[:5]
    
    return results

def main():
    trial_data = """- Trial 118: Quality 0.700 | Regimes: 4 | CV: 7.84 | EWMA=8+16 | states=4 | κ=11.6 | min_cov=1.0e-02
- Trial 119: Quality 0.475 | Regimes: 5 | CV: 1.95 | EWMA=12+16 | states=5 | κ=11.6 | min_cov=2.5e-03
- Trial 120: Quality 0.300 | Regimes: 4 | CV: 0.39 | EWMA=12+16 | states=4 | κ=1.6 | min_cov=5.0e-03
- Trial 121: Quality 0.307 | Regimes: 4 | CV: 0.44 | EWMA=12+16 | states=4 | κ=9.1 | min_cov=2.5e-03
- Trial 122: Quality 0.635 | Regimes: 5 | CV: 4.89 | EWMA=8+16 | states=5 | κ=9.1 | min_cov=1.0e-02
- Trial 123: Quality 0.418 | Regimes: 3 | CV: 1.26 | EWMA=8+24 | states=3 | κ=11.6 | min_cov=2.5e-03
- Trial 124: Quality 0.575 | Regimes: 5 | CV: 3.40 | EWMA=12+16 | states=5 | κ=1.6 | min_cov=5.0e-03
- Trial 125: Quality 0.408 | Regimes: 3 | CV: 1.17 | EWMA=8+24 | states=3 | κ=9.1 | min_cov=2.5e-03
- Trial 126: Quality 0.607 | Regimes: 3 | CV: 3.71 | EWMA=12+20 | states=3 | κ=1.6 | min_cov=1.0e-02
- Trial 127: Quality 0.543 | Regimes: 5 | CV: 2.90 | EWMA=8+20 | states=5 | κ=9.1 | min_cov=5.0e-03
- Trial 128: Quality 0.727 | Regimes: 5 | CV: 13.12 | EWMA=8+24 | states=5 | κ=1.6 | min_cov=2.5e-03
- Trial 129: Quality 0.312 | Regimes: 4 | CV: 0.48 | EWMA=12+16 | states=4 | κ=4.1 | min_cov=5.0e-03
- Trial 130: Quality 0.665 | Regimes: 3 | CV: 5.25 | EWMA=12+16 | states=3 | κ=4.1 | min_cov=2.5e-03
- Trial 131: Quality 0.293 | Regimes: 5 | CV: 0.37 | EWMA=12+16 | states=5 | κ=6.6 | min_cov=5.0e-03
- Trial 132: Quality 0.473 | Regimes: 5 | CV: 1.93 | EWMA=12+20 | states=5 | κ=6.6 | min_cov=1.0e-02
- Trial 133: Quality 0.470 | Regimes: 5 | CV: 1.90 | EWMA=12+20 | states=5 | κ=1.6 | min_cov=5.0e-03
- Trial 134: Quality 0.600 | Regimes: 3 | CV: 3.62 | EWMA=8+16 | states=3 | κ=4.1 | min_cov=1.0e-02
- Trial 135: Quality 0.665 | Regimes: 3 | CV: 5.25 | EWMA=12+16 | states=3 | κ=4.1 | min_cov=1.0e-02
- Trial 136: Quality 0.419 | Regimes: 3 | CV: 1.24 | EWMA=8+20 | states=3 | κ=9.1 | min_cov=2.5e-03
- Trial 137: Quality 0.586 | Regimes: 3 | CV: 3.30 | EWMA=12+20 | states=3 | κ=8.0 | min_cov=2.5e-03
- Trial 138: Quality 0.636 | Regimes: 5 | CV: 4.91 | EWMA=8+24 | states=5 | κ=0.2 | min_cov=1.0e-02
- Trial 139: Quality 0.545 | Regimes: 3 | CV: 2.65 | EWMA=12+20 | states=3 | κ=0.2 | min_cov=1.0e-02
- Trial 140: Quality 0.478 | Regimes: 5 | CV: 1.99 | EWMA=12+20 | states=5 | κ=8.0 | min_cov=2.5e-03
- Trial 141: Quality 0.477 | Regimes: 3 | CV: 1.82 | EWMA=8+20 | states=3 | κ=3.0 | min_cov=5.0e-03
- Trial 142: Quality 0.483 | Regimes: 5 | CV: 2.05 | EWMA=12+20 | states=5 | κ=3.0 | min_cov=1.0e-02
- Trial 143: Quality 0.560 | Regimes: 3 | CV: 2.87 | EWMA=12+20 | states=3 | κ=0.5 | min_cov=5.0e-03
- Trial 144: Quality 0.302 | Regimes: 4 | CV: 0.41 | EWMA=12+16 | states=4 | κ=3.0 | min_cov=2.5e-03
- Trial 145: Quality 0.588 | Regimes: 3 | CV: 3.38 | EWMA=8+16 | states=3 | κ=3.0 | min_cov=5.0e-03
- Trial 146: Quality 0.429 | Regimes: 3 | CV: 1.34 | EWMA=8+20 | states=3 | κ=8.0 | min_cov=1.0e-02"""
    
    # Parser les données
    trials = parse_trial_data(trial_data)
    print(f"Total des essais analysés: {len(trials)}")
    
    # Analyser les meilleurs performeurs
    results = analyze_by_criteria(trials)
    
    print("\n" + "="*80)
    print("TOP 5 DES MEILLEURS PERFORMEURS PAR QUALITÉ")
    print("="*80)
    for i, trial in enumerate(results['top_quality'], 1):
        print(f"{i}. Trial {trial['trial_number']}: Quality={trial['quality']:.3f} | "
              f"Regimes={trial['regimes']} | CV={trial['cv']:.2f} | "
              f"EWMA={trial['ewma_short']}+{trial['ewma_long']} | "
              f"states={trial['states']} | κ={trial['kappa']:.1f} | "
              f"min_cov={trial['min_cov']:.1e}")
    
    print("\n" + "="*80)
    print("TOP 5 DES PLUS STABLES (CV LE PLUS FAIBLE)")
    print("="*80)
    for i, trial in enumerate(results['lowest_cv'], 1):
        print(f"{i}. Trial {trial['trial_number']}: CV={trial['cv']:.2f} | "
              f"Quality={trial['quality']:.3f} | Regimes={trial['regimes']} | "
              f"EWMA={trial['ewma_short']}+{trial['ewma_long']} | "
              f"states={trial['states']} | κ={trial['kappa']:.1f}")
    
    print("\n" + "="*80)
    print("TOP 5 MEILLEUR ÉQUILIBRE QUALITÉ/STABILITÉ")
    print("="*80)
    for i, trial in enumerate(results['best_quality_cv_balance'], 1):
        balance_score = trial['quality'] / (trial['cv'] + 0.01)
        print(f"{i}. Trial {trial['trial_number']}: Score={balance_score:.2f} | "
              f"Quality={trial['quality']:.3f} | CV={trial['cv']:.2f} | "
              f"Regimes={trial['regimes']} | EWMA={trial['ewma_short']}+{trial['ewma_long']} | "
              f"states={trial['states']} | κ={trial['kappa']:.1f}")
    
    print("\n" + "="*80)
    print("ANALYSE DES PARAMÈTRES DES TOP PERFORMEURS")
    print("="*80)
    
    # Analyser les paramètres communs des top 10 par qualité
    top_10 = results['top_quality'][:10]
    
    print("\nDistribution des régimes dans le top 10:")
    regime_counts = {}
    for trial in top_10:
        regime_counts[trial['regimes']] = regime_counts.get(trial['regimes'], 0) + 1
    for regimes, count in sorted(regime_counts.items()):
        print(f"  {regimes} régimes: {count} essais")
    
    print("\nDistribution des EWMA dans le top 10:")
    ewma_counts = {}
    for trial in top_10:
        ewma_key = f"{trial['ewma_short']}+{trial['ewma_long']}"
        ewma_counts[ewma_key] = ewma_counts.get(ewma_key, 0) + 1
    for ewma, count in sorted(ewma_counts.items()):
        print(f"  EWMA {ewma}: {count} essais")
    
    print("\nDistribution des κ (kappa) dans le top 10:")
    kappa_ranges = {"faible (0-2)": 0, "moyen (2-6)": 0, "élevé (6+)": 0}
    for trial in top_10:
        if trial['kappa'] < 2:
            kappa_ranges["faible (0-2)"] += 1
        elif trial['kappa'] < 6:
            kappa_ranges["moyen (2-6)"] += 1
        else:
            kappa_ranges["élevé (6+)"] += 1
    for range_name, count in kappa_ranges.items():
        print(f"  κ {range_name}: {count} essais")

if __name__ == "__main__":
    main()