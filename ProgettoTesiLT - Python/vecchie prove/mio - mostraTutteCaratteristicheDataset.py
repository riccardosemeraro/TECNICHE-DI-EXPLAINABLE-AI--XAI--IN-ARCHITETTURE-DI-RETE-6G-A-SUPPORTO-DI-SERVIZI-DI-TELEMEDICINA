import pandas as pd
import shap
import matplotlib.pyplot as plt

from tqdm import tqdm
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Carico il dataset dal file csv
diabets_prediction = pd.read_csv('diabetes_prediction_dataset_v2.csv')

# Divido il dataset in due parti: una per le features e una per il target
X = diabets_prediction.drop(columns = 'diabetes', axis=1)
y = diabets_prediction['diabetes']

#Creazione dati di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=13)

# Costruzione del modello
#rf_clf = RandomForestClassifier(max_features=2, n_estimators =100 ,bootstrap = True)
rf_clf = RandomForestClassifier(max_features=10, n_estimators=50, max_depth=50, bootstrap=True, min_samples_leaf=10, criterion='gini')

rf_clf.fit(X_train, y_train) # Addestramento del modello

# Predizione del modello sui dati di test
y_pred = rf_clf.predict(X_test)

# voglio le feature
class_names = ['No Diabetes', 'Diabetes'] # risultati possibili
features = list(X_train.columns) # lista delle features

# Classificazione report
print(classification_report(y_test, y_pred))

## Parte 2: SHAP
shap.initjs() # per visualizzare i grafici SHAP
# Creo un oggetto TreeExplainer
explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_test)


# Visualizzazione dell'importanza delle features
figure = plt.figure()
plt.title("Variable Importance Plot - Global Interpretation")
shap.summary_plot(shap_values, X_test)
