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

from utils import minmax, rinormalizza, identifica_soglia_critica, risparmio, calcolo_metriche

#Lettura del dataset subito dopo pre-process
#diabets_prediction = pd.read_csv('diabetes_prediction_dataset_stringToInt.csv')
diabets_prediction = pd.read_csv('diabetes_prediction_dataset_v_senza_outliers.csv')

#prendo i valori massimi e minimi di ogni feature
minGend, maxGend, minAge, maxAge, minHyp, maxHyp, minHeart, maxHeart, minSmok, maxSmok, minBmi, maxBmi, minHbA1c, maxHbA1c, minBl, maxBl = minmax(diabets_prediction)

#Lettura del dataset dopo averlo normalizzato e rimosso outliers
diabets_prediction = pd.read_csv('diabetes_prediction_dataset_v2.csv')
#diabets_prediction.head()

print("---------------------------------------------------------------------------------------------------------------")

print("Numero di campioni: ", diabets_prediction.shape[0])

#data = diabets_prediction #da eliminare quando faccio under e over sampling

#voglio tutti i valori in float
data = diabets_prediction.astype(float)


#Oversampling spostato nel file changeDataset


#Split in train e test set
print("\n\n\n Divisione in train e test set")
x = data.drop(columns='diabetes')
#print(x)
y = data['diabetes']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,stratify=y,random_state=42)
print("Numero di campioni di test: ", ytest.shape[0])

#voglio importanre il modello RF salvato
rf = joblib.load('./content/random_forest_model_showData.pkl')

'''
#Random Forest
rf = RandomForestClassifier(max_features=10,n_estimators=150,max_depth=50,bootstrap=True)
rf.fit(xtrain,ytrain)
'''
ypred = rf.predict(xtest)
# Classification report
print(classification_report(ytest, ypred))

'''
classi_previste = rf.classes_
print("Classi previste dal modello:", classi_previste)

class_names = ['No Diabete', 'Diabete']

feature_names = list(xtrain.columns)
print("Feature names:", feature_names)


# Classification report
print(classification_report(ytest, ypred))



## Parte 2: SHAP
shap.initjs() # per visualizzare i grafici SHAP
# Creo un oggetto TreeExplainer
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(xtest) # Calcolo i valori SHAP per il test set
'''

class_names = ['No Diabete', 'Diabete']

feature_names = list(xtrain.columns)
print("Feature names:", feature_names)

#voglio importare il modello SHAP salvato
shap_values = np.load('./content/shap_values_showData.npy')

print("Shap Values 0:", shap_values[ :, :, 0].shape)
print("Shap Values 1:", shap_values[:,:,1].shape)
print("Xtest:", xtest.shape)

#---------------------------------------------------

# Visualizzazione del grafico SHAP Summary Plot
plt.figure()
plt.title("Summary Plot - Global Interpretation")
shap.summary_plot(shap_values[:,:,1], xtest, show=False)
savefig('./images/summary_plot_global_interpretation.png')

'''
#----------------SALVATAGGIO MODELLI------------------------------------------------------------------------------------

# Salvataggio del modello Random Forest
model_filename = './content/random_forest_model_showData.pkl'
joblib.dump(rf, model_filename)
print(f"Modello Random Forest salvato in {model_filename}")


# Salvataggio dei valori SHAP
shap_values_filename = './content/shap_values_showData.npy'
np.save(shap_values_filename, shap_values)
print(f"Valori SHAP salvati in {shap_values_filename}")

#---------------FINE SALVATAGGIO MODELLI----------------------------------------------------------------------------------
'''

#------- VARIABLE IMPORTANCE PLOT ----------------

# I valori shap_values sono una lista di array, uno per ogni classe (due in questo caso)
shap_values_class_0 = np.abs(shap_values[:,:,0]).mean(0) # Per la Classe 0
shap_values_class_1 = np.abs(shap_values[:,:,1]).mean(0) # Per la Classe 1

# Ordinamento delle feature in base all'importanza totale (somma dei contributi delle due classi)
indices = np.argsort(shap_values_class_0 + shap_values_class_1)

# Plot delle variabili con barre impilate per le due classi
plt.figure()

plt.barh(range(len(indices)), shap_values_class_1[indices], color='blue', label='Class 1 - Diabete')
#voglio posizionare la classe 0 a destra del grafico precedente
plt.barh(range(len(indices)), shap_values_class_0[indices], color='red', label='Class 0 - No Diabete', left=shap_values_class_1[indices])
plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)')
plt.yticks(range(len(indices)), np.array(feature_names)[indices])
plt.title('Variable importance plot - Global interpretation - RF')
#voglio la legenda in basso a destra
plt.legend(loc='lower right')
plt.tight_layout()
savefig('./images/variable_importance_plot_global_interpretation.png')
#--------------------------------------------------



#--------------------------------PER VEDERE FEATURES PIU IMPORTANTI --------------------------------
#----------------POI ANDRO' A FARE UNA FEATURE SELECTION CON TAGLI VERTICALI DEL DATASET------------
#---------------------------------------------PER VEDERE SE IL MODELLO MIGLIORA----------------------


#voglio acquisire le feature più importanti-------------------------------------------------------------------------
print("Feature importance")
importances = rf.feature_importances_ #importanza delle feature calcolata dal modello RF
indices = np.argsort(importances)[::-1] #ordino in modo decrescente le feature più importanti (da 0 a 7)
print(indices) #stampo gli indici delle feature più importanti
important_feature_name = []
for f in range(xtrain.shape[1]): #per tutte le features del dataset
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])) #stampo le feature più importanti
    print(feature_names[indices[f]]) #stampo i nomi delle feature più importanti
    if f <= 2: #prendo solo le prime 3 features più importanti
        important_feature_name.append(feature_names[indices[f]])

print("Features più importanti:", important_feature_name) #stampo le feature più importanti (le prime 3)
print ("Feature più importante:", important_feature_name[0]) #stampo la feature più importante


# VOGLIO STAMPARE I DEPENDENCE PLOT PER LE FEATURES PIU IMPORTANTI -----------------------------------------------------

#Rinormalizzo il dataset
dataset_rinormalizzato = rinormalizza(xtest, minGend, maxGend, minAge, maxAge, minHyp, maxHyp, minHeart, maxHeart, minSmok, maxSmok, minBmi, maxBmi, minHbA1c, maxHbA1c, minBl, maxBl)

print("Shape di shap_values:", shap_values[:,:,1].shape)
print("Shape di xtest:", xtest.shape)


#identifico i valori di soglia critica per le feature più importanti
max_changes_xtest = {}
feature_changes_dataset_rinormalizzato = {}
for feature in important_feature_name:
    max_change_xtest, feature_change = identifica_soglia_critica(feature, shap_values[:,:,1], xtest)
    max_change_dataset_rinormalizzato, feature_change_dataset_rinormalizzato  = identifica_soglia_critica(feature, shap_values[:, :, 1], dataset_rinormalizzato)

    max_changes_xtest[feature] = max_change_xtest
    feature_changes_dataset_rinormalizzato[feature] = feature_change_dataset_rinormalizzato

    print("Max change per ", feature, ":", max_change_xtest, " (Dati normalizzati)")
    print("Feature change per ", feature, ":", feature_change_dataset_rinormalizzato)

    # Salva in file
    #pulisco il file ogni volta che lo apro
    with open('./content/soglie_criticali.txt', 'w') as f:
        f.write("")

    with open('./content/soglie_criticali.txt', 'a') as f:
        f.write(f"Max change per '{feature}': {max_change_xtest :.2f} (Dati normalizzati)\n")
        f.write(f"Feature change per '{feature}': {feature_change_dataset_rinormalizzato :.2f}\n")

# Dependence plot per la feature più importante
#plt.figure(7)
shap.dependence_plot(important_feature_name[0], shap_values[:,:,1], xtest, interaction_index=important_feature_name[0], show=False)
plt.axvline(x=max_changes_xtest[important_feature_name[0]], color='red', linestyle='--', label=f'Soglia {important_feature_name[0]} = {max_changes_xtest[important_feature_name[0]]:.2f}')
plt.legend()
plt.title("Dependence Plot - Prima Feature più importante (Dati normalizzati)")
savefig('./images/dependence_plot_1_normalizzati.png')

#plt.figure(8)
shap.dependence_plot(important_feature_name[0], shap_values[:,:,1], dataset_rinormalizzato, feature_names=feature_names, interaction_index=important_feature_name[0], show=False)
plt.axvline(x=feature_changes_dataset_rinormalizzato[important_feature_name[0]], color='red', linestyle='--', label=f'Soglia {important_feature_name[0]} = {feature_changes_dataset_rinormalizzato[important_feature_name[0]]:.2f}')
plt.legend()
plt.title("Dependence Plot - Prima Feature più importante (Dati ripristinati)")
savefig('./images/dependence_plot_1_ripristinati.png')

# Dependence plot per la seconda feature più importante
#plt.figure(9)
shap.dependence_plot(important_feature_name[1], shap_values[:,:,1], xtest, feature_names=feature_names, interaction_index=important_feature_name[1], show=False)
plt.axvline(x=max_changes_xtest[important_feature_name[1]], color='red', linestyle='--', label=f'Soglia {important_feature_name[1]} = {max_changes_xtest[important_feature_name[1]]:.2f}')
plt.legend()
plt.title("Dependence Plot - Seconda feature più importante (Dati normalizzati)")
savefig('./images/dependence_plot_2_normalizzati.png')

#plt.figure(10)
shap.dependence_plot(important_feature_name[1], shap_values[:,:,1], dataset_rinormalizzato, feature_names=feature_names, interaction_index=important_feature_name[1], show=False)
plt.axvline(x=feature_changes_dataset_rinormalizzato[important_feature_name[1]], color='red', linestyle='--', label=f'Soglia {important_feature_name[1]} = {feature_changes_dataset_rinormalizzato[important_feature_name[1]]:.2f}')
plt.legend()
plt.title("Dependence Plot - Seconda feature più importante (Dati ripristinati)")
savefig('./images/dependence_plot_2_ripristinati.png')

# Dependence plot per la terza feature più importante
#plt.figure(11)
shap.dependence_plot(important_feature_name[2], shap_values[:,:,1], xtest, feature_names=feature_names, interaction_index=important_feature_name[2], show=False)
plt.axvline(x=max_changes_xtest[important_feature_name[2]], color='red', linestyle='--', label=f'Soglia {important_feature_name[2]} = {max_changes_xtest[important_feature_name[2]]:.2f}')
plt.legend()
plt.title("Dependence Plot - Terza feature più importante (Dati normalizzati)")
savefig('./images/dependence_plot_3_normalizzati.png')

#plt.figure(12)
shap.dependence_plot(important_feature_name[2], shap_values[:,:,1], dataset_rinormalizzato, feature_names=feature_names, interaction_index=important_feature_name[2], show=False)
plt.axvline(x=feature_changes_dataset_rinormalizzato[important_feature_name[2]], color='red', linestyle='--', label=f'Soglia {important_feature_name[2]} = {feature_changes_dataset_rinormalizzato[important_feature_name[2]]:.2f}')
plt.legend()
plt.title("Dependence Plot - Terza feature più importante (Dati ripristinati)")
savefig('./images/dependence_plot_3_ripristinati.png')


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

savefig('./images/decision_tree_rf.png')

#---------------------------------------------------------------------------------------------

plt.show()

#----------------------SCRITTURA SU FILE DELLE FEATURE PIU IMPORTANTI--------------------------------
#voglio salvare in un file il valore di important_feature_name
with open('important_feature_name.txt', 'w') as f:
    for item in important_feature_name:
        f.write("%s\n" % item)
#----------------------------------------------------------------------------------------------------





#----------------------ANALISI METRICHE-KPI-------------------------------------------------------------
print(" ")
dataset_originale = pd.read_csv('diabetes_prediction_dataset_v2.csv')
dataset_originale = rinormalizza(dataset_originale, minGend, maxGend, minAge, maxAge, minHyp, maxHyp, minHeart, maxHeart, minSmok, maxSmok, minBmi, maxBmi, minHbA1c, maxHbA1c, minBl, maxBl)

#Filtraggio
data = pd.DataFrame(dataset_originale)
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

#dra_originale,tempo_orignale,energia_orignale = calcolo_metriche(dataset_originale)
tempo_orignale,energia_orignale = calcolo_metriche(dataset_originale)
#print("DATA RATE - DATASET ORIGINALE: " + str(dra_originale) +" Mbps" )
print("TEMPO DI TRASMISSIONE - DATASET ORIGINALE: " + str(tempo_orignale) +" s")
print("ENERGIA UTILIZZATA - DATASET ORIGINALE: " + str(energia_orignale) + " j")

print(" ")

#dra_ridotto,tempo_ridotto,energia_ridotto = calcolo_metriche(dataset_ridotto)
tempo_ridotto,energia_ridotto = calcolo_metriche(dataset_ridotto)
#print("DATA RATE - DATASET RIDOTTO: " + str(dra_ridotto) +" Mbps" )
print("TEMPO DI TRASMISSIONE - DATASET RIDOTTO: " + str(tempo_ridotto) +" s")
print("ENERGIA UTILIZZATA - DATASET RIDOTTO: " + str(energia_ridotto) + " j")

print(" ")

#risparmio_dra = risparmio(dra_ridotto,dra_originale)
risparmio_tempo = risparmio(tempo_ridotto,tempo_orignale)
risparmio_energia = risparmio(energia_ridotto,energia_orignale)
#print("RISPARMIO DATA RATE: " + str(risparmio_dra) + " %")
print("RISPARMIO TEMPO DI TRASMISSIONE: " + str(risparmio_tempo) + " %")
print("RISPARMIO ENERGIA: " + str(risparmio_energia) + " %")