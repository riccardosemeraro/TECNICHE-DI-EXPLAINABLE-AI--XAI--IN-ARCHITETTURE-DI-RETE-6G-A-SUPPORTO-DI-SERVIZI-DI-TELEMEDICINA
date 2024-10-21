import pandas as pd
from IPython.core.display_functions import display
from sklearn.preprocessing import MinMaxScaler # per normalizzare i dati
from scipy import stats
import numpy as np

from utils import eliminazione, normalizzare, Outliers, rinormalizza, pre_process, under, over, conteggio

diabets_prediction = pd.read_csv('diabetes_prediction_dataset.csv')

#Lettura del dataset e pre-process

diabets_prediction.head()
print("Numero di campioni totali: ", diabets_prediction.shape[0])

diabets_prediction = pre_process(diabets_prediction)

#Rimozione Outliers
features = []
for feature in diabets_prediction.columns:
    if feature != 'diabetes':
        features.append(feature)
print(features)

for feature in features:
    diabets_prediction = Outliers(diabets_prediction, feature)

diabets_prediction.to_csv('diabetes_prediction_dataset_v_senza_outliers.csv', index=False)

#Normalizzazione
diabets_prediction, minGend, maxGend, minAge, maxAge, minHyp, maxHyp, minHeart, maxHeart, minSmok, maxSmok, minBmi, maxBmi, minHbA1c, maxHbA1c, minBl, maxBl = normalizzare(diabets_prediction)

#Over-sampling
'''
#Under-sampling
print("\n\n Under-sampling")
data=under(diabets_prediction)
conteggio(data)
'''

#Over-sampling
print("\n\n Over-sampling")
diabets_prediction=over(diabets_prediction)
conteggio(diabets_prediction)

diabets_prediction.to_csv('diabetes_prediction_dataset_vOver.csv', index=False)


print("Numero di campioni dopo la rimozione degli outliers: ", diabets_prediction.shape[0])

diabets_prediction.to_csv('diabetes_prediction_dataset_v2.csv', index=False)