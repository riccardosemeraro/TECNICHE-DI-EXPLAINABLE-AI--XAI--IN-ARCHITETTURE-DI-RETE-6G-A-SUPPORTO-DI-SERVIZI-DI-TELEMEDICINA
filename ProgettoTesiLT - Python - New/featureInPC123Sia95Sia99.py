#Importazione librerie

import pandas as pd
import numpy as np
from matplotlib.pyplot import savefig
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import tree
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler # per normalizzare i dati
from imblearn.over_sampling import RandomOverSampler

from utils import rinormalizza, risparmio, calcolo_metriche, Outliers, conteggio

#Lettura del dataset subito dopo pre-process
#diabets_prediction = pd.read_csv('diabetes_prediction_dataset_stringToInt.csv')
diabets_prediction = pd.read_csv('reduced_diabetes_prediction_features_with_target_cumulative_explained_95.csv') #reduced_diabetes_prediction_features_with_target_cumulative_explained_99.csv

#age, bmi, HbA1c_level, blood_glucose_level ---> Feature Presenti nel dataset

#prendo i valori massimi e minimi di ogni feature

minAge=min(diabets_prediction['age'])
maxAge=max(diabets_prediction['age'])

minBmi = min(diabets_prediction['bmi'])
maxBmi = max(diabets_prediction['bmi'])

minHbA1c = min(diabets_prediction['HbA1c_level'])
maxHbA1c = max(diabets_prediction['HbA1c_level'])

minBl = min(diabets_prediction['blood_glucose_level'])
maxBl = max(diabets_prediction['blood_glucose_level'])
#------------------------------------------------------------

#Rimozione Outliers
features = []
for feature in diabets_prediction.columns:
    if feature != 'diabetes':
        features.append(feature)
print(features)

for feature in features:
    diabets_prediction = Outliers(diabets_prediction, feature)

#----------------------------------------------------------

#Normalizzazione

cols_da_norm = list(range(diabets_prediction.shape[1]-1))
data_da_norm = diabets_prediction.iloc[:,cols_da_norm]
scal = MinMaxScaler()
data_norm = scal.fit_transform(data_da_norm)
diabets_prediction.iloc[:,cols_da_norm] = data_norm

#----------------------------------------------------------

data = diabets_prediction

#Oversampling
print("\n\nOver-sampling")
x = data.drop('diabetes', axis=1).values
y = data['diabetes'].values
over=RandomOverSampler(sampling_strategy='minority')
X_over, y_over = over.fit_resample(x, y)
columnsl=['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
df_over = pd.concat([pd.DataFrame(X_over, columns=columnsl),
pd.DataFrame(y_over, columns=['diabetes'])], axis=1)
data = df_over
conteggio(data)

print("Numero di campioni dopo la rimozione degli outliers: ", data.shape[0])


#----------------------------------------------------------
#Lettura del dataset dopo averlo normalizzato e rimosso outliers
print("Numero di campioni: ", data.shape[0])
data = data.astype(float)

#---------------------------------------------------------

#Split in train e test set
print("\n\n\n Divisione in train e test set")
x = data.drop(columns='diabetes')
#print(x)
y = data['diabetes']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,stratify=y,random_state=42)
print("Numero di campioni di test: ", ytest.shape[0])

#Random Forest
rf = RandomForestClassifier(max_features=10,n_estimators=150,max_depth=50,bootstrap=True)
rf.fit(xtrain,ytrain)
ypred = rf.predict(xtest)

classi_previste = rf.classes_
print("Classi previste dal modello:", classi_previste)

class_names = ['No Diabete', 'Diabete']

feature_names = list(xtrain.columns)
print("Feature names:", feature_names)


# Classification report
print(classification_report(ytest, ypred))


#Stampo l'albero di decisione del modello RF---------------------------------------------

# Estrai un albero dalla foresta (es. il primo)
estimator = rf.estimators_[0]

# Accedi alla struttura interna dell'albero
tree_structure = estimator.tree_

# Calcola la profondità massima dell'albero
max_depth = estimator.get_depth()  # Usa il metodo get_depth() per calcolare la profondità dell'albero
print(f"Profondità massima dell'albero: {max_depth}")

# Imposta la profondità che vuoi visualizzare (es. gli ultimi 2 livelli)
target_depth = max_depth - 1

# Imposta il grafico
plt.figure(num = 100, figsize=(50, 25) )  # Dimensione del grafico

# Disegna l'albero usando plot_tree
tree.plot_tree(estimator,
               feature_names=feature_names,
               class_names=class_names,
               filled=True,
               rounded=True,
               fontsize=12,
               max_depth=5) # Imposta la profondità dell'albero #target_depth

savefig('./images/decision_tree_rf-PC123.png')

#---------------------------------------------------------------------------------------------

plt.show()



#----------------------ANALISI METRICHE-KPI-------------------------------------------------------------
print(" ")
dataset_originale = pd.read_csv('diabetes_prediction_dataset_v2.csv')
#prendo i valori massimi e minimi di ogni feature

print(dataset_originale.columns)

minGend = min(dataset_originale['gender'])
maxGend = max(dataset_originale['gender'])
minAge = min(dataset_originale['age'])
maxAge = max(dataset_originale['age'])
minHyp = min(dataset_originale['hypertension'])
maxHyp = max(dataset_originale['hypertension'])
minHeart = min(dataset_originale['heart_disease'])
maxHeart = max(dataset_originale['heart_disease'])
minSmok = min(dataset_originale['smoking_history'])
maxSmok = max(dataset_originale['smoking_history'])
minBmi = min(dataset_originale['BMI'])
maxBmi = max(dataset_originale['BMI'])
minHbA1c = min(dataset_originale['HbA1C'])
maxHbA1c = max(dataset_originale['HbA1C'])
minBl = min(dataset_originale['glucose'])
maxBl = max(dataset_originale['glucose'])

dataset_originale = rinormalizza(dataset_originale, minGend, maxGend, minAge, maxAge, minHyp, maxHyp, minHeart, maxHeart, minSmok, maxSmok, minBmi, maxBmi, minHbA1c, maxHbA1c, minBl, maxBl)

#Filtraggio
dataset_ridotto = data #in questo caso non devo rimuovere nessuna riga o colonna perchè è il caso base
#print(data_ridotto)
diabetici = dataset_ridotto[dataset_ridotto['diabetes'] == 1]
print("Numero di Pazienti Diabetici: ", diabetici.shape[0])

#Calcolo delle Risorse
dimensione_in_byte_originale = dataset_originale.memory_usage(deep=True).sum()
print("Dimensione del Dataset ORIGINALE: ", dimensione_in_byte_originale, " byte")

dimensione_in_byte_ridotto = dataset_ridotto.memory_usage(deep=True).sum()
print("Dimensione del Dataset RIDOTTO: ", dimensione_in_byte_ridotto, " byte")

print(" ")

dra_originale,tempo_orignale,energia_orignale = calcolo_metriche(dataset_originale)
print("DATA RATE - DATASET ORIGINALE: " + str(dra_originale) +" Mbps" )
print("TEMPO DI TRASMISSIONE - DATASET ORIGINALE: " + str(tempo_orignale) +" s")
print("ENERGIA UTILIZZATA - DATASET ORIGINALE: " + str(energia_orignale) + " j")

print(" ")

dra_ridotto,tempo_ridotto,energia_ridotto = calcolo_metriche(dataset_ridotto)
print("DATA RATE - DATASET RIDOTTO: " + str(dra_ridotto) +" Mbps" )
print("TEMPO DI TRASMISSIONE - DATASET RIDOTTO: " + str(tempo_ridotto) +" s")
print("ENERGIA UTILIZZATA - DATASET RIDOTTO: " + str(energia_ridotto) + " j")

print(" ")

risparmio_dra = risparmio(dra_ridotto,dra_originale)
risparmio_tempo = risparmio(tempo_ridotto,tempo_orignale)
risparmio_energia = risparmio(energia_ridotto,energia_orignale)
print("RISPARMIO DATA RATE: " + str(risparmio_dra) + " %")
print("RISPARMIO TEMPO DI TRASMISSIONE: " + str(risparmio_tempo) + " %")
print("RISPARMIO ENERGIA: " + str(risparmio_energia) + " %")