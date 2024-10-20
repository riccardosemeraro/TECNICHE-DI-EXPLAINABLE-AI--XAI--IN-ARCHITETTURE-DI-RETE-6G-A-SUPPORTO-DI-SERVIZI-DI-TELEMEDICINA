from pyexpat import features

import pandas as pd
from IPython.core.display_functions import display
from sklearn.preprocessing import MinMaxScaler # per normalizzare i dati
from scipy import stats
import numpy as np

diabets_prediction = pd.read_csv('diabetes_prediction_dataset.csv')

features = []
for feature in diabets_prediction.columns:
    features.append(feature)
print(features)

#print(diabets_prediction['gender'].unique()) # per vedere i valori unici della colonna 'gender'

pd.set_option('future.no_silent_downcasting', True) # per evitare il warning 'FutureWarning: The default value of regex will change from True to False in a future version.'

# voglio riscrivere i dati della colonna 'gender': al posto di 'Female' metto 0, al posto 'Male' metto 1 e al posto di 'Other' metto -1
generi = {
    'Female': 0,
    'Male': 1,
    'Other': -1 }

diabets_prediction['gender'] = diabets_prediction['gender'].replace(generi) # sostituisco i valori della colonna

# voglio cancellare le intere righe dove la cella gender è uguale a Other (non ha senso per il mio modello)
# diabets_prediction = diabets_prediction[diabets_prediction['gender'] != 'Other']

# voglio riscrivere i dati della colonna 'smoking_history'
fumatore = {
    'never': 0, # 'mai fumato'
    'No Info': -1, # 'nessuna informazione'
    'current': 1, # 'fuma attualmente'
    'former': 2, # 'ex fumatore'
    'ever': 3, # 'fumato almeno una volta'
    'not current': 4 } # 'non fuma ora'

diabets_prediction['smoking_history'] = diabets_prediction['smoking_history'].replace(fumatore) # sostituisco i valori della colonna

# creo un nuovo file .csv con i dati modificati
diabets_prediction.to_csv('diabetes_prediction_dataset_v2.csv', index=False) #csv modificato dove sono state sostituite le stringhe con i valori numerici



# voglio normalizzare i dati con la funzione MinMaxScaler
scaler = MinMaxScaler()
diabets_prediction[features] = scaler.fit_transform(diabets_prediction[features])

# creo un nuovo file .csv con i dati modificati
diabets_prediction.to_csv('diabetes_prediction_dataset_v3.csv', index=False) #csv dove i dati sono stati normalizzati

#voglio eliminare gli outliers, ossia i valori che si trovano al di sopra dell'estremo superirore o al di sotto dell'estremo inferiore
# per farlo uso la funzione zscore di scipy.stats
z_scores = stats.zscore(diabets_prediction[features])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
diabets_prediction = diabets_prediction[filtered_entries]

# creo un nuovo file .csv con i dati modificati
diabets_prediction.to_csv('diabetes_prediction_dataset_v4.csv', index=False) #csv dove sono stati eliminati gli outliers


# voglio visualizzare i dati modificati
print(diabets_prediction)