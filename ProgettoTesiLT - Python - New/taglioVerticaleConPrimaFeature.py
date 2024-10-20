#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#----------------------------TAGLIO VERTICALE CON SOLO PRIMA FEATURE RILEVANTE------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

# Importo le librerie necessarie

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from utils import minmax, calcolo_metriche, risparmio, rinormalizza

#leggo dal file important_feature_name.txt i valori e li inserisco nella variabile important_feature_name
with open('important_feature_name.txt', 'r') as f:
    important_feature_name = f.read().splitlines()
print(important_feature_name) #stampo important_feature_name dalla fase precedente


#Lettura del dataset dopo averlo normalizzato e rimosso outliers
diabets_prediction = pd.read_csv('diabetes_prediction_dataset_v2.csv')

print("Numero di campioni: ", diabets_prediction.shape[0])

#voglio tutti i valori in float
data = diabets_prediction.astype(float)

#Split in train e test set
print("\n\n\n Divisione in train e test set")

print("Colonne di data:", list(data.columns))

data2 = data
for feature in data.columns:
    if feature not in important_feature_name[0]: #se la feature non è presente in important_feature_name la elimino
        #print("Elimino la feature: ", feature)
        data2 = data2.drop(columns=feature)
#x = data.drop(columns='diabetes') #inclusa nel for precedente
x=data2
print("Colonne di X:", list(x.columns))

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






#----------------------ANALISI METRICHE-KPI-------------------------------------------------------------

dataset_non_normalizzato = pd.read_csv('diabetes_prediction_dataset_v_senza_outliers.csv')

#prendo i valori massimi e minimi di ogni feature
minGend, maxGend, minAge, maxAge, minHyp, maxHyp, minHeart, maxHeart, minSmok, maxSmok, minBmi, maxBmi, minHbA1c, maxHbA1c, minBl, maxBl = minmax(dataset_non_normalizzato)

print(" ")
dataset_originale = pd.read_csv('diabetes_prediction_dataset_v2.csv')
dataset_originale = rinormalizza(dataset_originale, minGend, maxGend, minAge, maxAge, minHyp, maxHyp, minHeart, maxHeart, minSmok, maxSmok, minBmi, maxBmi, minHbA1c, maxHbA1c, minBl, maxBl)

#Filtraggio
data = pd.DataFrame(dataset_originale) #creo un nuovo dataset con i valori del dataset originale per poterlo filtrare senza modificare l'originale
#Rimuovo le feature meno importanti lasciando solo le prime 3 più importanti
for feature in data.columns:
    if feature not in important_feature_name[0]: #se la feature non è presente in important_feature_name la elimino
        if feature != 'diabetes':
            data = data.drop(columns=feature)

dataset_ridotto =  data # (solo la prima feature più importante) + (target)

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
