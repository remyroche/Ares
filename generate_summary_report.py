import pandas as pd
import json
from datetime import datetime

# Charger le fichier CSV final
csv_file = './outcomes/all_trials_results_ETHUSDT_20251110_222817_final.csv'
df = pd.read_csv(csv_file)

# Charger le fichier JSON des résultats HPO
json_file = './artifacts/rolling_hmm_hpo_results.json'
with open(json_file, 'r') as f:
    hpo_results = json.load(f)

# Extraire les meilleurs paramètres
best_params = hpo_results['data']['best_params']
best_score = hpo_results['data']['best_score']

# Trouver l'essai qui utilise la configuration finale
final_trial = None
for idx, row in df.iterrows():
    if row['Final_Configuration'] == 'Yes':
        final_trial = row
        break

# Générer le rapport résumé
report = f"""
# Rapport d'optimisation HPO pour le modèle HMM roulant
Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Résumé de l'optimisation
- Nombre d'essais évalués : {len(df)}
- Meilleur score obtenu : {-best_score:.4f}
- Nombre d'essais avec le meilleur score : {len(df[df['Quality_Score'] == -best_score])}

## Meilleurs paramètres trouvés
- EWMA Config: {best_params.get('ewma_config_idx', 'N/A')}
- n_components: {best_params.get('n_components', 'N/A')}
- min_covar: {best_params.get('min_covar', 'N/A')}
- kappa: {best_params.get('kappa', 'N/A')}

## Configuration finale sauvegardée comme artefact
La configuration finale sauvegardée comme artefact utilise les meilleurs paramètres trouvés lors de l'optimisation HPO.

"""
if final_trial is not None:
    report += f"""
### Essai utilisant la configuration finale
- Numéro d'essai : {final_trial['Trial']}
- Rang : {final_trial['Rank']}
- Score de qualité : {final_trial['Quality_Score']:.4f}
- Nombre de régimes : {final_trial['N_Regimes']}
- CV : {final_trial['Within_CV']:.2f} (W:{final_trial['Within_CV_Std']:.2f} B:{final_trial['Between_CV']:.2f})
- Temporal : {final_trial['Temporal_Smoothness']:.4f}
- Balance : {final_trial['Balance_Score']:.4f}
"""

# Ajouter les 10 meilleurs essais
report += """
## Top 10 des essais
"""
top_trials = df.head(10)
for idx, row in top_trials.iterrows():
    report += f"""
{row['Rank']}. Essai {row['Trial']} - Score : {row['Quality_Score']:.4f}
   - Paramètres : K={row['K']}, Kappa={row['Kappa']}, N_Mixtures={row['N_Mixtures']}
   - Configuration finale : {'Oui' if row['Final_Configuration'] == 'Yes' else 'Non'}
"""

# Sauvegarder le rapport
with open('./outcomes/hpo_summary_report.md', 'w') as f:
    f.write(report)

print("Rapport résumé sauvegardé sous : ./outcomes/hpo_summary_report.md")
print("\n" + report)
