import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Cargar modelo y escalador
modelo = joblib.load('modelo_logistic_regression.pkl')
scaler = joblib.load('scaler_final.pkl')

# Configuración inicial
st.set_page_config(page_title="Predicción de Fuga de Clientes", page_icon="📉")
st.title("📱 Predicción de Fuga en Clientes de Telecomunicaciones")
st.markdown("Completa la información del cliente para estimar su probabilidad de fuga.")

st.markdown("---")

# Distribución en columnas
col1, col2 = st.columns(2)

with col1:
    falla_llamada = st.number_input("📞 ¿Según su historial, cuántas fallas ha tenido hasta el momento?", min_value=0.0, step=1.0)
    segundos_uso = st.number_input("⏱️ Cuántos segundos pasa en llamadas", min_value=0.0, step=1.0)
    sms_dia = st.number_input("📨 Promedio de SMS por día", min_value=0.0, step=0.1)

with col2:
    monto_cobrado = st.selectbox("💰 ¿Del 0 a 9 es un cliente que genera valor? (0=bajo, 9=alto)", list(range(10)))
    frecuencia_uso = st.number_input("📊 ¿Según su historial, cuántas llamadas ha tenido hasta el momento?", min_value=0.0, step=1.0)
    quejas_mes = st.number_input("😠 ¿Cuántas quejas se han registrado hasta el momento?", min_value=0.0, step=0.01)

# Botón para predecir
if st.button("🔍 Predecir Fuga"):
    # Armar vector de entrada
    entrada = np.array([[falla_llamada, monto_cobrado, segundos_uso, frecuencia_uso, sms_dia, quejas_mes]])
    entrada_esc = scaler.transform(entrada)
    prob_fuga = modelo.predict_proba(entrada_esc)[0, 1]

    # Tabla de resumen
    resumen = pd.DataFrame(entrada, columns=[
        "Falla de Llamada", "Monto Cobrado", "Segundos de Uso", 
        "Frecuencia de Uso", "SMS al Día", "Quejas por Mes"
    ])
    st.markdown("#### 📋 Resumen del Cliente:")
    st.dataframe(resumen, use_container_width=True)

    # Resultado
    st.markdown("---")
    st.markdown(f"### 🎯 Probabilidad estimada de fuga: **{prob_fuga:.2%}**")
    st.progress(prob_fuga)

    # Interpretación del riesgo
    if prob_fuga >= 0.8:
        st.error("🔴 **Alto riesgo de fuga.** Se recomienda actuar de inmediato con ofertas, llamadas o beneficios.")
    elif prob_fuga >= 0.5:
        st.warning("🟠 **Riesgo moderado.** Es recomendable hacer seguimiento al cliente.")
    else:
        st.success("🟢 **Bajo riesgo.** El cliente parece estar satisfecho con el servicio.")

    st.markdown("---")
    st.markdown("📌 **Nota:** Esta predicción está basada en un modelo de regresión logística entrenado con datos históricos.")
