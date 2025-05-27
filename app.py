import os
import sys
import subprocess
import importlib.util

if importlib.util.find_spec("gdown") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
import gdown
from joblib import load
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource(show_spinner="Mengunduh dan memuat model CVD Presence...")
def load_model_presence():
    model_path_presence = "model_presence.joblib"
    # Unduh model_presence menggunakan gdrive
    url = "https://drive.google.com/uc?id=1nWUhcG4Uyotk_zbvi_LfnchPcCKgAp9f"
    if not os.path.exists(model_path_presence):
        gdown.download(url, model_path_presence, quiet=False)
    return load(model_path_presence)
# Load model
model_presence = load_model_presence()
model_risk = load("modeling/model_risk.joblib")

# Load encoders
result_target_presence = load("modeling/encoder_target.joblib")
encoder_Hypertension_presence = load("modeling/encoder_presence_Hypertension.joblib")
encoder_ECG_Abnormality_presence = load("modeling/encoder_presence_ECG_Abnormality.joblib")
encoder_Diabetes_presence = load("modeling/encoder_presence_Diabetes.joblib")
encoder_Alcohol_presence = load("modeling/encoder_presence_Alcohol.joblib")
encoder_Previous_Stroke_presence = load("modeling/encoder_presence_Previous_Stroke.joblib")
encoder_Family_History_presence = load("modeling/encoder_presence_Family_History.joblib")
encoder_CVD_Risk_Score_presence = load("modeling/encoder_presence_CVD_Risk_Score.joblib")

result_target_risk = load("modeling/encoder_target_risk.joblib")
encoder_Hypertension_risk = load("modeling/encoder_risk_Hypertension.joblib")
encoder_ECG_Abnormality_risk = load("modeling/encoder_risk_ECG_Abnormality.joblib")
encoder_Diabetes_risk = load("modeling/encoder_risk_Diabetes.joblib")
encoder_Alcohol_risk = load("modeling/encoder_risk_Alcohol.joblib")
encoder_Previous_Stroke_risk = load("modeling/encoder_risk_Previous_Stroke.joblib")
encoder_Family_History_risk = load("modeling/encoder_risk_Family_History.joblib")

# Load scaler
scaler_Insulin_Resistance_presence = load("modeling/scaler_presence_Insulin_Resistance.joblib")
scaler_Pulse_Pressure_presence = load("modeling/scaler_presence_Pulse_Pressure.joblib")
scaler_Diastolic_BP_presence = load("modeling/scaler_presence_Diastolic_BP.joblib")
scaler_Systolic_BP_presence = load("modeling/scaler_presence_Systolic_BP.joblib")
scaler_Resting_HR_presence = load("modeling/scaler_presence_Resting_HR.joblib")

scaler_Insulin_Resistance_risk = load("modeling/scaler_risk_Insulin_Resistance.joblib")
scaler_Pulse_Pressure_risk = load("modeling/scaler_risk_Pulse_Pressure.joblib")
scaler_Diastolic_BP_risk = load("modeling/scaler_risk_Diastolic_BP.joblib")
scaler_Systolic_BP_risk = load("modeling/scaler_risk_Systolic_BP.joblib")
scaler_Resting_HR_risk = load("modeling/scaler_risk_Resting_HR.joblib")

def data_preprocessing_risk(data):
    """Preprocessing data

    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction

    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    data = data.copy()
    df = pd.DataFrame()

    # Encode numeric features
    df['Insulin_Resistance'] = scaler_Insulin_Resistance_risk.transform(np.asarray(data['Insulin_Resistance']).reshape(-1, 1))[:, 0]
    df['Pulse_Pressure'] = scaler_Pulse_Pressure_risk.transform(np.asarray(data['Pulse_Pressure']).reshape(-1, 1))[:, 0]
    df['Diastolic_BP'] = scaler_Diastolic_BP_risk.transform(np.asarray(data['Diastolic_BP']).reshape(-1, 1))[:, 0]
    df['Systolic_BP'] = scaler_Systolic_BP_risk.transform(np.asarray(data['Systolic_BP']).reshape(-1, 1))[:, 0]
    df['Resting_HR'] = scaler_Resting_HR_risk.transform(np.asarray(data['Resting_HR']).reshape(-1, 1))[:, 0]
    

    # Encode categorical features
    df['Hypertension'] = encoder_Hypertension_risk.transform(data['Hypertension'])
    df['ECG_Abnormality'] = encoder_ECG_Abnormality_risk.transform(data['ECG_Abnormality'])
    df['Diabetes'] = encoder_Diabetes_risk.transform(data['Diabetes'])
    df['Alcohol'] = encoder_Alcohol_risk.transform(data['Alcohol'])
    df['Previous_Stroke'] = encoder_Previous_Stroke_risk.transform(data['Previous_Stroke'])
    df['Family_History'] = encoder_Family_History_risk.transform(data['Family_History'])

    return df

def prediction_risk(data):
    """Making prediction
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data
 
    Returns:
        str: Prediction result (Low, Moderate or High)
    """
    result_risk = model_risk.predict(data)
    final_result_risk = result_target_risk.inverse_transform(result_risk)[0]
    return final_result_risk

def data_preprocessing_presence(data):
    """Preprocessing data

    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction

    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    data = data.copy()
    df = pd.DataFrame()

    # Encode numeric features
    df['Insulin_Resistance'] = scaler_Insulin_Resistance_presence.transform(np.asarray(data['Insulin_Resistance']).reshape(-1, 1))[:, 0]
    df['Pulse_Pressure'] = scaler_Pulse_Pressure_presence.transform(np.asarray(data['Pulse_Pressure']).reshape(-1, 1))[:, 0]
    df['Diastolic_BP'] = scaler_Diastolic_BP_presence.transform(np.asarray(data['Diastolic_BP']).reshape(-1, 1))[:, 0]
    df['Systolic_BP'] = scaler_Systolic_BP_presence.transform(np.asarray(data['Systolic_BP']).reshape(-1, 1))[:, 0]
    df['Resting_HR'] = scaler_Resting_HR_presence.transform(np.asarray(data['Resting_HR']).reshape(-1, 1))[:, 0]
    
    # Encode categorical features
    df['Hypertension'] = encoder_Hypertension_presence.transform(data['Hypertension'])
    df['ECG_Abnormality'] = encoder_ECG_Abnormality_presence.transform(data['ECG_Abnormality'])
    df['Diabetes'] = encoder_Diabetes_presence.transform(data['Diabetes'])
    df['Alcohol'] = encoder_Alcohol_presence.transform(data['Alcohol'])
    df['Previous_Stroke'] = encoder_Previous_Stroke_presence.transform(data['Previous_Stroke'])
    df['Family_History'] = encoder_Family_History_presence.transform(data['Family_History'])
    df['CVD_Risk_Score'] = encoder_CVD_Risk_Score_presence.transform(data['CVD_Risk_Score'])

    return df    

def prediction_presence(data):
    """Making prediction
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data
 
    Returns:
        str: Prediction result (No or Yes)
    """
    result_present = model_presence.predict(data)
    final_result_present = result_target_presence.inverse_transform(result_present)[0]
    return final_result_present


# Streamlit
# Tampilan Utama
st.sidebar.markdown(
    """
    <style>
       [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #f0f8ff, #e0f7fa);
    }
    </style>
""", unsafe_allow_html=True
)

st.sidebar.title("📘 Informasi Aplikasi")
st.sidebar.info(
    """
Aplikasi ini membantu mendeteksi risiko penyakit kardiovaskular (CVD) secara dini
berdasarkan data kesehatan dasar.\n

🔍 **Penting**:
- Pastikan mengisi data dengan benar.\n
- Aplikasi ini hanya untuk tujuan informasi dan **bukan** menggantikan konsultasi medis profesional.\n
- Hasil prediksi bersifat indikatif dan harus dikonfirmasi dengan dokter. 
"""
)

st.sidebar.header("📌 Cara Menggunakan Aplikasi")
st.sidebar.info(
    """
1️⃣ Masukkan data kesehatan sesuai kolom yang tersedia.\n
2️⃣ Pilih tab **Prediksi CVD Risk Score** atau **Prediksi CVD Presence**.\n
3️⃣ Klik tombol **Prediksi** untuk melihat hasil.\n
4️⃣ Gunakan hasil prediksi sebagai referensi awal, tetapi tetap konsultasikan dengan dokter profesional.
"""
)

st.sidebar.header("👨‍💻 Tentang Pengembang")
st.sidebar.markdown("[🔗 Kunjungi LinkedIn Saya](https://www.linkedin.com/in/silvia-rachmawati/)")

st.write("This is a web application to predict CVD using Machine Learning.")
st.write("Please fill in the following form to make a prediction.")

tab_risk, tab_presence = st.tabs(["Prediksi CVD Risk Score", "Prediksi CVD Presence"])

with tab_risk:
    st.subheader("Prediksi Risiko CVD")
    st.write("Silakan isi data berikut untuk memprediksi risiko CVD.")

    # Input Form
    data_risk = pd.DataFrame({
        "Hypertension" : [st.selectbox("Hypertension", options=encoder_Hypertension_risk.classes_, index=1, key="Hypertension_risk")],
        "ECG_Abnormality" : [st.selectbox("ECG Abnormality", options=encoder_ECG_Abnormality_risk.classes_, index=2, key="ECG_Abnormality_risk")],
        "Diabetes" : [st.selectbox("Diabetes", options=encoder_Diabetes_risk.classes_, index=1, key="Diabetes_risk")],
        "Alcohol" : [st.selectbox("Alcohol", options=encoder_Alcohol_risk.classes_, index=1, key="Alcohol_risk")],
        "Previous_Stroke" : [st.selectbox("Previous Stroke", options=encoder_Previous_Stroke_risk.classes_, index=1, key="Previous_Stroke_risk")],
        "Family_History" : [st.selectbox("Family History", options=encoder_Family_History_risk.classes_, index=1, key="Family_History_risk")],
        "Insulin_Resistance" : [st.number_input("Insulin Resistance", value=4.732879, step=0.1, key="Insulin_Resistance_risk")],
        "Pulse_Pressure" : [st.number_input("Pulse Pressure", value=42.972956, step=0.1, key="Pulse_Pressure_risk")],
        "Diastolic_BP" : [st.number_input("Diastolic BP", value=86.808942, step=0.1, key="Diastolic_BP_risk")],
        "Systolic_BP" : [st.number_input("Systolic BP", value=111.648090, step=0.1, key="Systolic_BP_risk")],
        "Resting_HR" : [st.number_input("Resting HR", value=72.329284, step=0.1, key="Resting_HR_risk")]
    })
    # Button to make prediction
    st.subheader("Data yang dimasukkan")
    st.dataframe(data_risk, width=800)
    
    # Button to make prediction
    if st.button("Prediksi", key="predict_risk"):
        st.write("Memproses data...")
      # Preprocess data
        new_data_risk = data_preprocessing_risk(data=data_risk)
        with st.expander("View the Preprocessed Data"):
            st.dataframe(data=new_data_risk, width=800, height=10)
        st.write("Data telah diproses, sekarang memprediksi risiko CVD...")
        st.write("Mohon tunggu sebentar...")

        # Make prediction
        result_risk = prediction_risk(new_data_risk)
        st.session_state.hasil_prediksi_risk = result_risk  # Save result in session state

    if "hasil_prediksi_risk" in st.session_state:
        result_risk = st.session_state.hasil_prediksi_risk
       # Display result
        st.subheader("Hasil Prediksi")
        st.write(f"Prediksi CVD Risiko: {result_risk}")
        if result_risk == "High":
            st.error("⚠️ Risiko Tinggi! Segera konsultasikan ke dokter.")
        elif result_risk == "Moderate":
            st.warning("⚠️ Risiko Sedang. Waspada dan mulai ubah gaya hidup.")
        else:
            st.success("✅ Risiko Rendah. Tetap jaga pola hidup sehat.")

        st.markdown("---")
        st.subheader("📌 Rekomendasi Gaya Hidup Sehat")
        st.markdown("""
                    - 🏃 Rutin berolahraga minimal 30 menit sehari
                    - 🥦 Konsumsi makanan rendah lemak dan garam
                    - 🚭 Hindari merokok dan konsumsi alkohol berlebihan
                    - 💤 Tidur cukup 7-8 jam sehari
                    - 🩺 Rutin cek kesehatan dan konsultasi dokter
                    - 🧘‍♂️ Kelola stres dengan baik
                    - 💧 Minum cukup air putih setiap hari
                    - 🥗 Perbanyak konsumsi buah dan sayur
                    """)
        st.markdown("🧠 *Prediksi ini hanya bersifat indikatif, bukan diagnosis medis.*")

        if st.button("Reset", key="reset_risk"):
            st.session_state.hasil_prediksi_risk = None
            st.experimental_rerun()

# Tab untuk Prediksi Keberadaan CVD
with tab_presence:
    st.subheader("Prediksi Keberadaan CVD")
    st.write("Silakan isi data berikut untuk memprediksi keberadaan CVD.")
    
    # Input Form
    data_presence = pd.DataFrame({
        "Hypertension" : [st.selectbox("Hypertension", options=encoder_Hypertension_presence.classes_, index=1, key="Hypertension_presence")],
        "ECG_Abnormality" : [st.selectbox("ECG Abnormality", options=encoder_ECG_Abnormality_presence.classes_, index=0, key="ECG_Abnormality_presence")],
        "Diabetes" : [st.selectbox("Diabetes", options=encoder_Diabetes_presence.classes_, index=1, key="Diabetes_presence")],
        "Alcohol" : [st.selectbox("Alcohol", options=encoder_Alcohol_presence.classes_, index=0, key="Alcohol_presence")],
        "Previous_Stroke" : [st.selectbox("Previous Stroke", options=encoder_Previous_Stroke_presence.classes_, index=1, key="Previous_Stroke_presence")],
        "Family_History" : [st.selectbox("Family History", options=encoder_Family_History_presence.classes_, index=1, key="Family_History_presence")],
        "CVD_Risk_Score" : [st.selectbox("CVD Risk Score", options=encoder_CVD_Risk_Score_presence.classes_, index=1, key="CVD_Risk_Score_presence")],
        "Insulin_Resistance" : [st.number_input("Insulin Resistance", value=6.441541, step=0.1, key="Insulin_Resistance_presence")],
        "Pulse_Pressure" : [st.number_input("Pulse Pressure", value=33.438115, step=0.1, key="Pulse_Pressure_presence")],
        "Diastolic_BP" : [st.number_input("Diastolic BP", value=73.272788, step=0.1, key="Diastolic_BP_presence")],
        "Systolic_BP" : [st.number_input("Systolic BP", value=116.245744, step=0.1, key="Systolic_BP_presence")],
        "Resting_HR" : [st.number_input("Resting HR", value=60.970855, step=0.1, key="Resting_HR_presence")]
     })
    # Button to make prediction
    st.subheader("Data yang dimasukkan")
    st.dataframe(data_presence, width=800)
    
    if st.button("Prediksi", key="predict_presence"):
        st.write("Memproses data...")
        # Preprocess data
        new_data_presence = data_preprocessing_presence(data=data_presence)
        with st.expander("View the Preprocessed Data"):
            st.dataframe(data=new_data_presence, width=800, height=10)
        st.write("Data telah diproses, sekarang memprediksi keberadaan CVD...")
        st.write("Mohon tunggu sebentar...")
        
        # Make prediction
        result_presence = prediction_presence(new_data_presence)
        st.session_state.hasil_prediksi_presence = result_presence
    if "hasil_prediksi_presence" in st.session_state:
        result_presence = st.session_state.hasil_prediksi_presence

        # Display result
        st.subheader("Hasil Prediksi")
        st.write(f"Prediksi CVD Presence: {result_presence}")
        if result_presence == "Yes":
         st.error("CVD Presence: **Terdeteksi**")
        else:
         st.success("CVD Presence: **Tidak Terdeteksi**")
    
        if st.button("Reset", key="reset_presence"):
            st.session_state.hasil_prediksi_presence = None
            st.experimental_rerun()
