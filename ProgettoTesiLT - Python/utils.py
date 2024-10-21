from sklearn.preprocessing import MinMaxScaler # per normalizzare i dati
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from scipy import stats
import matplotlib.colors as mcolors



#-----------------------------------------------------------------------------------------------------------------------
#---Preprocessing, normalizzazione e ripristino-------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

def pre_process(diabets_prediction):
    pd.set_option('future.no_silent_downcasting', True)  # per evitare il warning 'FutureWarning: The default value of regex will change from True to False in a future version.'

    # voglio riscrivere i dati della colonna 'gender': al posto di 'Female' metto 0, al posto 'Male' metto 1 e al posto di 'Other' metto -1
    generi = {
        'Female': 0,
        'Male': 1,
        'Other': 0.5}

    diabets_prediction['gender'] = diabets_prediction['gender'].replace(generi)  # sostituisco i valori della colonna


    # voglio cancellare le intere righe dove la cella gender è uguale a Other (non ha senso per il mio modello)
    # diabets_prediction = diabets_prediction[diabets_prediction['gender'] != 'Other']

    # voglio riscrivere i dati della colonna 'smoking_history'
    fumatore = {
        'never': 0,  # 'mai fumato'
        'No Info': 0.2,  # 'nessuna informazione'
        'current': 0.4,  # 'fuma attualmente'
        'former': 0.6,  # 'ex fumatore'
        'ever': 0.8,  # 'fumato almeno una volta'
        'not current': 1}  # 'non fuma ora'

    diabets_prediction['smoking_history'] = diabets_prediction['smoking_history'].replace(fumatore)  # sostituisco i valori della colonna

    diabets_prediction.to_csv('diabetes_prediction_dataset_stringToInt.csv', index=False)

    return diabets_prediction

def eliminazione(data):
    num_zeros = (data.iloc[:, :-1] == 0).sum(axis=1)
    data = data[num_zeros < 3]
    return data

def normalizzare(dataframe):
    minGend, maxGend, minAge, maxAge, minHyp, maxHyp, minHeart, maxHeart, minSmok, maxSmok, minBmi, maxBmi, minHbA1c, maxHbA1c, minBl, maxBl = minmax(dataframe)

    cols_da_norm = list(range(dataframe.shape[1]-1))
    data_da_norm = dataframe.iloc[:,cols_da_norm]
    scal = MinMaxScaler()
    data_norm = scal.fit_transform(data_da_norm)
    dataframe.iloc[:,cols_da_norm] = data_norm

    ##dataframe['gender'] = (dataframe['gender'] - minGend) / (maxGend - minGend)
    #dataframe['age'] = (dataframe['age'] - minAge) / (maxAge - minAge)
    ##dataframe['hypertension'] = (dataframe['hypertension'] - minHyp) / (maxHyp - minHyp)
    ##dataframe['heart_disease'] = (dataframe['heart_disease'] - minHeart) / (maxHeart - minHeart)
    #dataframe['smoking_history'] = (dataframe['smoking_history'] - minSmok) / (maxSmok - minSmok)
    #dataframe['bmi'] = (dataframe['bmi'] - minBmi) / (maxBmi - minBmi)
    #dataframe['HbA1c_level'] = (dataframe['HbA1c_level'] - minHbA1c) / (maxHbA1c - minHbA1c)
    #dataframe['blood_glucose_level'] = (dataframe['blood_glucose_level'] - minBl) / (maxBl - minBl)


    return (dataframe, minGend, maxGend, minAge, maxAge, minHyp, maxHyp, minHeart, maxHeart, minSmok, maxSmok, minBmi, maxBmi, minHbA1c, maxHbA1c, minBl, maxBl)

def Outliers(df, feature):
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    if feature == 'hypertension' or feature == 'heart_disease':
        lower = 0
        upper = 1

    df_clean = df[(df[feature] >= lower) & (df[feature] <= upper)]

    #print(feature + " ", df_clean[feature].unique())

    return df_clean

def minmax(data):

    #['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

    minGend=min(data['gender'])
    maxGend=max(data['gender'])

    minAge=min(data['age'])
    maxAge=max(data['age'])

    minHyp=min(data['hypertension'])
    maxHyp = max(data['hypertension'])

    minHeart = min(data['heart_disease'])
    maxHeart = max(data['heart_disease'])

    minSmok= min(data['smoking_history'])
    maxSmok = max(data['smoking_history'])

    minBmi = min(data['bmi'])
    maxBmi = max(data['bmi'])

    minHbA1c = min(data['HbA1c_level'])
    maxHbA1c = max(data['HbA1c_level'])

    minBl = min(data['blood_glucose_level'])
    maxBl = max(data['blood_glucose_level'])

    return (minGend, maxGend, minAge, maxAge, minHyp, maxHyp, minHeart, maxHeart, minSmok, maxSmok, minBmi, maxBmi, minHbA1c, maxHbA1c, minBl, maxBl)

def rinormalizza(data, minGend, maxGend, minAge, maxAge, minHyp, maxHyp, minHeart, maxHeart, minSmok, maxSmok, minBmi, maxBmi, minHbA1c, maxHbA1c, minBl, maxBl):

    data2 = data.copy()

    feature_names = list(data.columns)

    '''
    data['gender'] = (data['gender']) * (maxGend - minGend)+minGend
    data['age'] = (data['age']) * (maxAge - minAge)+minAge
    data['hypertension'] = (data['hypertension']) * (maxHyp - minHyp)+minHyp
    data['heart_disease'] = (data['heart_disease']) * (maxHeart - minHeart)+minHeart
    data['smoking_history'] = (data['smoking_history']) * (maxSmok - minSmok)+minSmok
    data['bmi'] = (data['bmi']) * (maxBmi - minBmi)+minBmi
    data['HbA1c_level'] = (data['HbA1c_level']) * (maxHbA1c - minHbA1c)+minHbA1c
    data['blood_glucose_level'] = (data['blood_glucose_level']) * (maxBl - minBl)+minBl
    '''

    data2[feature_names[0]] = (data[feature_names[0]]) * (maxGend - minGend)+minGend
    data2[feature_names[1]] = (data[feature_names[1]]) * (maxAge - minAge)+minAge
    data2[feature_names[2]] = (data[feature_names[2]]) * (maxHyp - minHyp)+minHyp
    data2[feature_names[3]] = (data[feature_names[3]]) * (maxHeart - minHeart)+minHeart
    data2[feature_names[4]] = (data[feature_names[4]]) * (maxSmok - minSmok)+minSmok
    data2[feature_names[5]] = (data[feature_names[5]]) * (maxBmi - minBmi)+minBmi
    data2[feature_names[6]] = (data[feature_names[6]]) * (maxHbA1c - minHbA1c)+minHbA1c
    data2[feature_names[7]] = (data[feature_names[7]]) * (maxBl - minBl)+minBl

    return(data2)


def identifica_soglia_critica(feature, shap_values_1, xtest):
    """
    Identifica la soglia critica per una data feature basata sui valori SHAP.

    Parameters:
    - feature (str): Nome della feature.
    - shap_values_1 (np.array): Valori SHAP.
    - xtest (pd.DataFrame): Dataset di test.

    Returns:
    - critical_value (float): Valore critico identificato.
    """
    # Controlla la dimensione dei dati
    #print(f"Shape of xtest[feature]: {xtest[feature].shape}")
    #print(f"Shape of SHAP values for feature '{feature}': {shap_values_1[:, xtest.columns.get_loc(feature)].shape}")

    '''
    # Crea un DataFrame con la feature e i valori SHAP
    df_shap = pd.DataFrame({
        feature: xtest[feature],
        'SHAP': shap_values_1[:, xtest.columns.get_loc(feature)]  # Valori SHAP solo per la feature corrente
    })

    # Ordina il DataFrame in base alla feature
    df_shap_sorted = df_shap.sort_values(by=feature)

    # Calcola la derivata prima dei valori SHAP rispetto alla feature
    df_shap_sorted['delta_SHAP'] = df_shap_sorted['SHAP'].diff() # Derivata prima

    # Identifica il punto di massimo cambiamento
    critical_idx = df_shap_sorted['delta_SHAP'].abs().idxmax()
    critical_value = df_shap_sorted.loc[critical_idx, feature]
    '''


    #PROVA 2 CORRETTAA
    # Crea un DataFrame con la feature e i valori SHAP
    df_shap = pd.DataFrame({
        feature: xtest[feature],
        'SHAP': shap_values_1[:, xtest.columns.get_loc(feature)]
    })

    # Ordina il DataFrame in base alla feature
    df_shap_sorted = df_shap.sort_values(by=feature)

    # Calcola il cambiamento assoluto nei valori SHAP
    df_shap_sorted['SHAP_change'] = df_shap_sorted['SHAP'].abs().diff()

    # Trova il punto di massimo cambiamento
    max_change_idx = df_shap_sorted['SHAP_change'].idxmax()

    #critical_value = df_shap_sorted.loc[max_change_idx, feature]
    max_change = df_shap_sorted.loc[max_change_idx, 'SHAP_change'] #SHAP_change

    #voglio inserire dove è presente il nan il valore 0
    df_shap_sorted['SHAP_change'] = df_shap_sorted['SHAP_change'].fillna(0)

    maxFeature = max(df_shap_sorted[feature]) #valore massimo della feature
    minFeature = min(df_shap_sorted[feature]) #valore minimo della feature

    feature_change = max_change * (maxFeature - minFeature) + minFeature #normalizzo il valore

    '''
    #PROVA CON I COLORI (INCOMPLETA)
    cmap = plt.get_cmap('RdBu')  # Cambia secondo necessità
    #colors = cmap(shap_values_1)
    #colors = np.array([cmap(normalized_shap[:, feature_idx]) for feature_idx in range(shap_values.shape[1])]).transpose(1, 0, 2)  # Shape: (516, 8, 4)
    print("COLOR SHAPE")
    print(colors.shape)
    purplish_indices = [i for i, color in enumerate(colors) if is_purplish(color)]

    if purplish_indices:
        # Trova il valore SHAP minimo che soddisfa la condizione
        threshold_shap = shap_values_1[purplish_indices].min()
        print(f"Il valore di soglia SHAP oltre il quale il colore diventa violaceo è: {threshold_shap}")
    else:
        print("Nessun valore SHAP soddisfa la condizione per essere considerato violaceo.")

    max_change = threshold_shap # Valore SHAP critico

    maxFeature = max(xtest[feature])  # valore massimo della feature
    minFeature = min(xtest[feature])  # valore minimo della feature

    feature_change = max_change * (maxFeature - minFeature) + minFeature  # normalizzo il valore
    '''
    return max_change, feature_change

def is_purplish(color, threshold=0.5):
    # Controlla quanti componenti ha il colore
    if len(color) == 4:
        r, g, b, a = color  # RGBA
    elif len(color) == 3:
        r, g, b = color  # RGB
    else:
        raise ValueError(f"Formato di colore non riconosciuto: {color}")
    # Definisci una condizione per considerare un colore come violaceo
    # Ad esempio, se rosso e blu sono entrambi sopra una certa soglia e verde è basso
    return r > threshold and b > threshold and g < threshold

#-----------------------------------------------------------------------------------------------------------------------
#---Under-sampling, over-sampling e conteggio dei campioni--------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

def under(data):
    positivi = data[data.iloc[:, -1] == 1]
    size=positivi.shape[0]
    negativi = data[data.iloc[:, -1] == 0].sample(n=size, random_state=42)
    #print(negativi.shape[0])
    #print(positivi.shape[0])
    data=pd.concat([positivi, negativi])
    return(data)

def over(data):
    x = data.drop('diabetes', axis=1).values
    y = data['diabetes'].values
    over=RandomOverSampler(sampling_strategy='minority')
    X_over, y_over = over.fit_resample(x, y)
    columnsl=['gender','age','hypertension','heart_disease','smoking_history','BMI','HbA1C','glucose']
    df_over = pd.concat([pd.DataFrame(X_over, columns=columnsl),
    pd.DataFrame(y_over, columns=['diabetes'])], axis=1)
    #print(df_over)
    return(df_over)

def conteggio(data):
    conteggio_diabete = len(data[data['diabetes'] == 1])
    print("Numero di campioni con diabete (1):", conteggio_diabete)
    conteggio_nodiabete = len(data[data['diabetes'] == 0])
    print("Numero di campioni senza diabete (0):", conteggio_nodiabete)
'''
def underdataset1(data):
    positivi = data[data.iloc[:, -1] == 1]
    size = positivi.shape[0]
    negativi = data[data.iloc[:, -1] == 0].sample(n=size, random_state=42)
    # print(negativi.shape[0])
    # print(positivi.shape[0])
    data = pd.concat([positivi, negativi])
    return (data)

def overdataset1(data):
    x = data.drop('diabetes', axis=1).values
    y = data['Outcome'].values
    over=RandomOverSampler(sampling_strategy='minority')
    X_over, y_over = over.fit_resample(x, y)
    columnsl=['Glucose','BMI', 'Insulin', 'BloodPressure',
    'DiabetesPedigreeFunction', 'Pregnancies', 'SkinThickness','Age']
    df_over = pd.concat([pd.DataFrame(X_over, columns=columnsl),
    pd.DataFrame(y_over, columns=['Outcome'])], axis=1)
    print(df_over)
    return(df_over)

def conteggiodataset1(data):
    conteggio_diabete = len(data[data['Outcome'] == 1])
    print("Numero di campioni con diabete (1):", conteggio_diabete)
    conteggio_nodiabete = len(data[data['Outcome'] == 0])
    print("Numero di campioni senza diabete (0):", conteggio_nodiabete)
'''
#-----------------------------------------------------------------------------------------------------------------------
#---Specifiche del dataset----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

def calcolo_metriche(data):
    dr=100 # [Mbps] data rate riportato nel file come specifica
    datarate=125000 #per convertire da Mbps a byte/sec
    costante=1*(1e-12) # per convertire da pico j a j
    original_size=(data.shape[0])
    #dra=original_size*dr
    dim_originale = data.memory_usage(deep=True).sum()
    bit=dim_originale*8 #da byte a bit
    tempo=dim_originale/(dr*datarate)
    energia=bit*costante
    #return(dra,tempo,energia)
    return(tempo, energia)

def risparmio(red,orig):
    prop=(red*100)/orig
    perc=100-prop
    return(perc)






#Scrittura per salvataggio dati
def scrittura(file, commento, contenuto=None):
    with open(file, 'a') as file: # Apri il file in modalità append
        file.write(commento + "\n")
        if contenuto is not None:
            file.write(str(contenuto) + "\n")
    return ()
