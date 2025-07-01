# 📱 Predicción de Fuga de Clientes en Telecomunicaciones

Este proyecto busca predecir la probabilidad de que un cliente de telecomunicaciones abandone la empresa, utilizando técnicas de análisis de datos, ingeniería de características, selección de variables, modelado y despliegue web con Streamlit.

---

## 📘 Diccionario de Datos

| Columna                    | Explicación                                                                                                                            |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------|
| Falla de Llamada           | Número de fallas en llamadas                                                                                                           |
| Quejas                     | Variable binaria (0: Sin queja, 1: Con queja)                                                                                          |
| Tiempo de Suscripción      | Total de meses de suscripción del cliente                                                                                              |
| Monto Cobrado              | Atributo ordinal (0: monto más bajo, 9: monto más alto)                                                                                |
| Segundos de Uso            | Total de segundos utilizados en llamadas                                                                                               |
| Frecuencia de Uso          | Número total de llamadas realizadas                                                                                                    |
| Frecuencia de SMS          | Número total de mensajes de texto enviados                                                                                             |
| Números Distintos Llamados | Número total de números telefónicos distintos contactados                                                                              |
| Grupo de Edad              | Atributo ordinal (1: grupo etario más joven, 5: grupo etario más adulto)                                                               |
| Plan Tarifario             | Variable binaria (1: Prepago, 2: Contrato)                                                                                             |
| Estado                     | Variable binaria (1: Activo, 2: Inactivo)                                                                                              |
| Edad                       | Edad del cliente                                                                                                                       |
| Valor del Cliente          | Valor calculado del cliente                                                                                                            |
| Fuga                       | Etiqueta de clase (1: cliente se fue, 0: cliente se mantiene)                                                                          |

---

## 🔍 Objetivo

Identificar con anticipación a los clientes con mayor riesgo de fuga para tomar decisiones preventivas y mejorar la retención. El modelo está basado en comportamiento de uso, quejas y características del servicio.

---

## 🧠 Tecnologías utilizadas

- Python 3
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit
- Joblib
- Git + GitHub

---

## ⚙️ Flujo del proyecto

### 1. Exploración y limpieza de datos
- Se eliminaron duplicados y se trató con valores extremos usando IQR (winsorización).
- Se cambiaron nombres de columnas para mejor comprensión.
- Se revisaron estadísticas y se visualizaron los datos.

### 2. Ingeniería de características
Se generaron nuevas variables a partir de las existentes:
- `fallas_por_mes`
- `llamadas_al_día`
- `sms_al_día`
- `quejas_por_mes`
> Esto permitió captar mejor el comportamiento de los clientes.

### 3. Selección de variables
Se aplicó `RFECV` con regresión logística para elegir las variables más importantes.  
Las seleccionadas fueron:
- Falla de Llamada  
- Monto Cobrado  
- Segundos de Uso  
- Frecuencia de Uso  
- sms_al_día  
- quejas_por_mes

### 4. Modelado
- División en datos de entrenamiento y prueba (`train_test_split`).
- Balanceo de clases con **SMOTE**.
- Normalización con `StandardScaler`.
- Modelo final: **Regresión Logística**.
- Métricas obtenidas: Accuracy, Precision, Recall.

### 5. Despliegue con Streamlit
- Se creó una aplicación web donde el usuario puede ingresar los datos de un cliente usando sliders.
- La app devuelve la **probabilidad de fuga** y una alerta según el nivel de riesgo.

