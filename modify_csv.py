import pandas as pd
import json

# Charger le fichier CSV
csv_file = './outcomes/all_trials_results_ETHUSDT_20251110_222817.csv'
df = pd.read_csv(csv_file)

# Charger le fichier JSON
json_file = './artifacts/rolling_hmm_hpo_results.json'
with open(json_file, 'r') as f:
    data = json.load(f)

# Extraire les résultats des essais
coarse_results = data['data']['coarse_results']
fine_results = data['data'].get('fine_results', [])
refinement_results = data['data'].get('refinement_results', [])

# Combiner tous les résultats
all_results = coarse_results + fine_results + refinement_results

# Créer un dictionnaire pour mapper les numéros d'essai aux paramètres
trial_params = {}
for i, result in enumerate(all_results, 1):
    trial_params[i] = result['params']

# Mettre à jour les colonnes du CSV
for idx, row in df.iterrows():
    trial_num = row['Trial']
    if trial_num in trial_params:
        params = trial_params[trial_num]
        
        # Mettre à jour les colonnes avec les valeurs des paramètres
        df.at[idx, 'K'] = params.get('n_components', 'N/A')
        df.at[idx, 'Kappa'] = params.get('kappa', 'N/A')
        df.at[idx, 'N_Mixtures'] = params.get('n_components', 'N/A')
        
        # Pour le HMM roulant, nous n'avons pas ces paramètres
        df.at[idx, 'Base_Alpha'] = 'N/A'
        df.at[idx, 'Learning_Rate'] = 'N/A'
        df.at[idx, 'SVI_Iterations'] = 'N/A'
        df.at[idx, 'ELBO'] = 'N/A'

# Sauvegarder le fichier CSV modifié
output_file = './outcomes/all_trials_results_ETHUSDT_20251110_222817_modified.csv'
df.to_csv(output_file, index=False)

print(f"Fichier CSV modifié sauvegardé sous : {output_file}")
