import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV, RFE
import joblib

# ---------- CONFIGURACIÓN GLOBAL DE ESTILO ---------- #
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16
})

# ---------- FUNCIONES ---------- #
def cargar_datos(ruta_csv):
    return pd.read_csv(ruta_csv)

def winsorizar_iqr(df, columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    df[columna] = df[columna].apply(lambda x: lim_inf if x < lim_inf else lim_sup if x > lim_sup else x)
    return df

def renombrar_columnas(df):
    df.rename(columns={
        'Call  Failure': 'Falla de Llamada',
        'Complains': 'Quejas',
        'Subscription  Length': 'Tiempo de Suscripción',
        'Charge  Amount': 'Monto Cobrado',
        'Seconds of Use': 'Segundos de Uso',
        'Frequency of use': 'Frecuencia de Uso',
        'Frequency of SMS': 'Frecuencia de SMS',
        'Distinct Called Numbers': 'Números Distintos Llamados',
        'Age Group': 'Grupo de Edad',
        'Tariff Plan': 'Plan Tarifario',
        'Status': 'Estado',
        'Age': 'Edad',
        'Customer Value': 'Valor del Cliente',
        'Churn': 'Fuga'
    }, inplace=True)
    return df

def crear_nuevas_variables(df):
    df['fallas_por_mes'] = df['Falla de Llamada'] / df['Tiempo de Suscripción']
    df['llamadas_al_día'] = (df['Frecuencia de Uso'] / df['Tiempo de Suscripción']) * 30
    df['sms_al_día'] = (df['Frecuencia de SMS'] / df['Tiempo de Suscripción']) * 30
    df['quejas_por_mes'] = df['Quejas'] / df['Tiempo de Suscripción']
    return df

def escalar_datos(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def seleccionar_variables_RFECV(X_train, y_train, min_features=3):
    clf = LogisticRegression(max_iter=1000, solver='liblinear')
    cv = StratifiedKFold(5)
    rfecv = RFECV(estimator=clf, step=1, cv=cv, scoring='accuracy', min_features_to_select=min_features, n_jobs=2)
    rfecv.fit(X_train, y_train)
    return rfecv

def aplicar_RFE_por_fold(X, y, n_features):
    clf = LogisticRegression(max_iter=1000, solver='liblinear')
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        selector = RFE(clf, n_features_to_select=n_features)
        selector.fit(X.iloc[train_idx], y.iloc[train_idx])
        print(f"Fold {i+1}: {list(X.columns[selector.support_])}")

def guardar_objetos(objetos_dict):
    for nombre, objeto in objetos_dict.items():
        joblib.dump(objeto, f'{nombre}.pkl')

def guardar_datasets(df_total, df_train, df_test, ruta_base):
    df_total.to_csv(f'{ruta_base}/data_procesada.csv', index=False)
    df_train.to_csv(f'{ruta_base}/data_procesada_train.csv', index=False)
    df_test.to_csv(f'{ruta_base}/data_procesada_test.csv', index=False)

# ---------- PIPELINE PRINCIPAL ---------- #
def pipeline_feature_engineering(ruta_csv, ruta_guardado):
    data = cargar_datos(ruta_csv)
    data = data.drop_duplicates()
    data = renombrar_columnas(data)

    variables_outliers = ['Falla de Llamada','Tiempo de Suscripción',
       'Segundos de Uso', 'Frecuencia de Uso', 'Frecuencia de SMS',
       'Números Distintos Llamados','Valor del Cliente']
    for var in variables_outliers:
        data = winsorizar_iqr(data, var)

    data = crear_nuevas_variables(data)
    data.drop(columns='Edad', inplace=True)

    X = data.drop(columns='Fuga')
    y = data['Fuga']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, scaler = escalar_datos(X_train, X_test)

    rfecv = seleccionar_variables_RFECV(X_train_scaled, y_train)

    print(f"Número óptimo de características: {rfecv.n_features_}")
    print("Variables seleccionadas:", list(X_train.columns[rfecv.support_]))

    aplicar_RFE_por_fold(X, y, rfecv.n_features_)

    X_final = X.loc[:, rfecv.support_]
    final_df = pd.concat([X_final, y], axis=1)

    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        final_df.drop(columns='Fuga'), final_df['Fuga'], stratify=final_df['Fuga'], test_size=0.2, random_state=42)

    guardar_datasets(final_df,
                     pd.concat([X_train_final, y_train_final], axis=1),
                     pd.concat([X_test_final, y_test_final], axis=1),
                     ruta_guardado)

    guardar_objetos({'scaler': scaler, 'rfecv': rfecv})

# Ejecutar (ajustar rutas según necesidad)
# pipeline_feature_engineering('ruta/a/customer_churn.csv', 'ruta/a/guardar/resultados')