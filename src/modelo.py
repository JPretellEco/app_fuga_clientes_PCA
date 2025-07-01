import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# ---------- FUNCIÓN DE ENTRENAMIENTO Y PREDICCIÓN ---------- #
def entrenar_y_predecir(ruta_train, ruta_test):
    # Cargar datos
    data_train = pd.read_csv(ruta_train)
    data_test = pd.read_csv(ruta_test)

    # Separar variables
    X = data_train.drop(columns='Fuga')
    y = data_train['Fuga']

    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, Y_resampled = smote.fit_resample(X, y)

    # Separar conjunto de validación
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, Y_resampled,
                                                      stratify=Y_resampled,
                                                      test_size=0.2, random_state=42)

    # Escalamiento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Entrenar modelo
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluación en validación
    y_val_pred = model.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    conf_matrix = confusion_matrix(y_val, y_val_pred)

    print("\n--- EVALUACIÓN EN VALIDACIÓN ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print("Matriz de Confusión:")
    print(conf_matrix)

    # Aplicar transformación al test y predecir
    X_test = data_test.drop(columns='Fuga')
    y_test = data_test['Fuga']
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = model.predict(X_test_scaled)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)

    print("\n--- EVALUACIÓN EN TEST ---")
    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")

    # Guardar modelo y scaler
    joblib.dump(model, 'modelo_logistic_regression.pkl')
    joblib.dump(scaler, 'scaler_final.pkl')

# Ejemplo de uso:
entrenar_y_predecir('E:/Mi unidad/VSC/proyecto2/data/procesada1/data_procesada_train.csv','E:/Mi unidad/VSC/proyecto2/data/procesada1/data_procesada_test.csv')