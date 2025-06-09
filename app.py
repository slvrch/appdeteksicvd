import os
import pandas as pd
import json
import gdown
from joblib import load 
import numpy as np
import streamlit as st
import requests


with open("features_order_risk.json", "r") as f:
    features_order_risk = json.load(f)

with open("features_order_presence.json", "r") as f:
    features_order_presence = json.load(f)

with open("label_options_all.json", "r") as f:
    label_options = json.load(f)

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

# Inisialisasi session state untuk menyimpan status registrasi
if 'registered' not in st.session_state:
    st.session_state['registered'] = False

# Cek query parameter untuk auto-set session jika URL mengandung ?registered=true
query_params = st.query_params
if query_params.get("registered") == ["true"]:
    st.session_state['registered'] = True
    
# Jika belum terdaftar, tampilkan form registrasi
if not st.session_state['registered']:
    st.title("Form Registrasi Pengguna")
    nama = st.text_input("Nama")
    email = st.text_input("Email")
    no_tlp = st.text_input("Nomor WhatsApp")

    if st.button("Submit"):
        if not nama or not email or not no_tlp:
            st.warning("Mohon lengkapi semua kolom.")
        else:
            try:
                response = requests.post("https://fastapicvd-production.up.railway.app/register", json={
                    "nama": nama,
                    "email": email,
                    "no_tlp": no_tlp
                })
                if response.status_code == 200:
                    st.success("Registrasi berhasil!")
                    st.session_state['registered'] = True
                    st.session_state['nama'] = nama
                    st.session_state['email'] = email
                    st.session_state['no_tlp'] = no_tlp
                    st.query_params = {"registered": "true"}
                    st.rerun()                   
                else:
                    st.error("Registrasi gagal: "+ response.text)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghubungi server: {e}")

# Jika sudah terdaftar, tampilkan dua tab prediksi
if st.session_state['registered']:
    st.title("Menu Prediksi CVD")
    st.write("Please fill in the following form to make a prediction.")
    
    tabs = st.tabs(["Prediksi CVD Risk Score", "Prediksi CVD Presence"])

   #=============================================================TAB 1================================================================================ 
    with tabs[0]:
        st.subheader("Masukkan Data untuk Prediksi risiko CVD")
        st.write("Form untuk prediksi risiko akan ditampilkan di sini.")

        # Input Form
        data_risk = {
        "Hypertension": 
            st.selectbox(
                "Hypertension", 
                label_options["risk"]["Hypertension"], 
                key="Hypertension_risk",
                help="Tekanan darah tinggi, kondisi di mana tekanan darah dalam arteri meningkat secara kronis. Pilih 'Yes' jika ada riwayat hipertensi, 'No' jika tidak."
            ),
        "ECG_Abnormality":
            st.selectbox(
                "ECG Abnormality",
                label_options["risk"]["ECG_Abnormality"], 
                key="ECG_Abnormality_risk",
                help="Kelainan pada hasil elektrokardiogram (ECG), yang dapat menunjukkan masalah jantung. Pilih 'Normal' jika tidak ada kelainan; 'Arrhythmia' jika ketidaknormalan dalam irama jantung, berupa detak jantung terlalu cepat, terlalu lambat, atau tidak teratur; 'Ischemia' jika kondisi otot jantung tidak mendapatkan cukup oksigen karena aliran darah berkurang."
            ),
        "Diabetes":
            st.selectbox(
                "Diabetes",
                label_options["risk"]["Diabetes"],
                key="Diabetes_risk",
                help="Kondisi di mana tubuh tidak dapat mengatur kadar gula darah dengan baik. Pilih 'Yes' jika ada riwayat diabetes, 'No' jika tidak."
            ),
        "Alcohol":
            st.selectbox(
                "Alcohol", 
                label_options["risk"]["Alcohol"],
                key="Alcohol_risk",
                help="Konsumsi alkohol, yang dapat mempengaruhi kesehatan jantung. Pilih 'Yes' jika mengonsumsi alkohol, 'No' jika tidak."
            ),
        "Previous_Stroke":
            st.selectbox(
                "Previous Stroke", 
                label_options["risk"]["Previous_Stroke"], 
                key="Previous_Stroke_risk",
                help="Riwayat stroke sebelumnya, yang dapat meningkatkan risiko CVD. Pilih 'Yes' jika ada riwayat stroke, 'No' jika tidak."
            ),
        "Family_History":
            st.selectbox(
                "Family History", 
                label_options["risk"]["Family_History"],
                key="Family_History_risk",
                help="Riwayat penyakit jantung dalam keluarga, yang dapat meningkatkan risiko CVD. Pilih 'Yes' jika ada riwayat keluarga, 'No' jika tidak."
            ),
        "Insulin_Resistance":
            st.number_input(
                "Insulin Resistance", 
                step=0.1, 
                key="Insulin_Resistance_risk",
                help="Kondisi dimana sel-sel dalam tubuh tidak merespon dengan baik terhadap insulin. Tingkat resistensi insulin terjadi saat glukosa tidak bisa masuk ke dalam sel dengan efisien yang mengakibatkan peningkatan kadar glukosa dalam darah sehingga dapat mempengaruhi risiko CVD. Nilai normal berkisar antara 0 hingga 10, dengan nilai lebih tinggi menunjukkan resistensi yang lebih besar."
            ),
        "Pulse_Pressure":
            st.number_input(
                "Pulse Pressure", 
                step=0.1, 
                key="Pulse_Pressure_risk",
                help="Perbedaan antara tekanan sistolik dan diastolik, yang dapat menunjukkan kesehatan jantung. Nilai normal berkisar antara 30 hingga 50 mmHg."
            ),
        "Diastolic_BP":
            st.number_input(
                "Diastolic BP",
                step=0.1, 
                key="Diastolic_BP_risk",
                help="Tekanan darah diastolik, yaitu tekanan pada arteri ketika jantung berada dalam kondisi istirahat di antara dua detak, ketika jantung mengisi kembali dengan darah. Nilai normal berkisar antara 60 hingga 80 mmHg."
            ),
        "Systolic_BP":
            st.number_input(
                "Systolic BP", 
                step=0.1, 
                key="Systolic_BP_risk",
                help="Tekanan darah sistolik, yaitu tekanan pada arteri ketika jantung berkontraksi dan memompa darah keluar ke seluruh tubuh. Nilai normal berkisar antara 90 hingga 120 mmHg."
            ),
        "Resting_HR":
            st.number_input(
                "Resting HR", 
                step=0.1, 
                key="Resting_HR_risk",
                help="Jumlah denyut jantung per menit ketika seseorang dalam kondisi istirahat penuh, yang dapat menunjukkan kesehatan jantung. Nilai normal berkisar antara 60 hingga 100 detak per menit."
            )
     }
        # Button to make prediction
        if st.button("Prediksi", key="predict_risk"):
            try:
                ordered_data_risk = {key: data_risk[key] for key in features_order_risk}

                if any(val == "" for val in ordered_data_risk.values()):
                    st.warning("Mohon lengkapi semua kolom sebelum melakukan prediksi.")
                    st.stop()
                    
                response = requests.post(
                    "https://fastapicvd-production.up.railway.app/predict-risk",
                    json=ordered_data_risk
                )
                if response.status_code == 200:
                    result_risk = response.json()["prediction_risk"]
                    st.session_state.hasil_prediksi_risk = result_risk
                    st.success(f"Prediksi berhasil! Hasil: {result_risk}")

                    # Kirim hasil ke Supabase
                    try:
                        requests.post("https://fastapicvd-production.up.railway.app/save-prediction",
                         json={
                             "nama": st.session_state.get("nama"),
                             "email": st.session_state.get("email"),
                             "no_tlp": st.session_state.get("no_tlp"),
                             "target": "risk",
                             "hasil_prediksi": result_risk
                         })
                    except Exception as err:
                        st.warning(f"Gagal menyimpan hasil ke database: {err}")
                
                else:
                    st.error("Gagal memproses prediksi: " + response.text)
            
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghubungi server: {e}")
                    
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
                del st.session_state["hasil_prediksi_risk"]
                st.rerun()

    #================================================================================TAB 2===============================================================================
    # Tab untuk Prediksi Keberadaan CVD
    with tabs[1]:
        st.subheader("Masukkan Data untuk Prediksi Keberadaan CVD")
        st.write("Form untuk prediksi keberadaan CVD akan ditampilkan di sini.")

        # Input Form
        data_presence = {
        "Hypertension":
            st.selectbox(
                "Hypertension", 
                label_options["presence"]["Hypertension"], 
                key="Hypertension_presence",
                help="Tekanan darah tinggi, kondisi di mana tekanan darah dalam arteri meningkat secara kronis. Pilih 'Yes' jika ada riwayat hipertensi, 'No' jika tidak."
            ),
        "ECG_Abnormality":
            st.selectbox(
                "ECG Abnormality",
                label_options["presence"]["ECG_Abnormality"],
                key="ECG_Abnormality_presence",
                help="Kelainan pada hasil elektrokardiogram (ECG), yang dapat menunjukkan masalah jantung. Pilih 'Normal' jika tidak ada kelainan; 'Arrhythmia' jika ketidaknormalan dalam irama jantung, berupa detak jantung terlalu cepat, terlalu lambat, atau tidak teratur; 'Ischemia' jika kondisi otot jantung tidak mendapatkan cukup oksigen karena aliran darah berkurang."
            ),
        "Diabetes":
            st.selectbox(
                "Diabetes",
                label_options["presence"]["Diabetes"], 
                key="Diabetes_presence",
                help="Kondisi di mana tubuh tidak dapat mengatur kadar gula darah dengan baik. Pilih 'Yes' jika ada riwayat diabetes, 'No' jika tidak."
            ),
        "Alcohol":
            st.selectbox(
                "Alcohol", 
                label_options["presence"]["Alcohol"], 
                key="Alcohol_presence",
                help="Konsumsi alkohol, yang dapat mempengaruhi kesehatan jantung. Pilih 'Yes' jika mengonsumsi alkohol, 'No' jika tidak."
            ),
        "Previous_Stroke":
            st.selectbox(
                "Previous Stroke", 
                label_options["presence"]["Previous_Stroke"],
                key="Previous_Stroke_presence",
                help="Riwayat stroke sebelumnya, yang dapat meningkatkan risiko CVD. Pilih 'Yes' jika ada riwayat stroke, 'No' jika tidak."
            ),
        "Family_History":
            st.selectbox(
                "Family History", 
                label_options["presence"]["Family_History"], 
                key="Family_History_presence",
                help="Riwayat penyakit jantung dalam keluarga, yang dapat meningkatkan risiko CVD. Pilih 'Yes' jika ada riwayat keluarga, 'No' jika tidak."
            ),
        "CVD_Risk_Score":
            st.selectbox(
                "CVD Risk Score", 
                label_options["presence"]["CVD_Risk_Score"], 
                key="CVD_Risk_Score_presence",
                help="Skor tingkat risiko seseorang untuk mengembangkan penyakit kardiovaskular di masa depan. Pilih 'Low' jika skor risiko rendah, 'Moderate' jika sedang, dan 'High' jika tinggi."
            ),
        "Insulin_Resistance":
            st.number_input(
                "Insulin Resistance", 
                step=0.1, 
                key="Insulin_Resistance_presence",
                help="Kondisi dimana sel-sel dalam tubuh tidak merespon dengan baik terhadap insulin. Tingkat resistensi insulin terjadi saat glukosa tidak bisa masuk ke dalam sel dengan efisien yang mengakibatkan peningkatan kadar glukosa dalam darah sehingga dapat mempengaruhi risiko CVD. Nilai normal berkisar antara 0 hingga 10, dengan nilai lebih tinggi menunjukkan resistensi yang lebih besar."
            ),
        "Pulse_Pressure":
            st.number_input(
                "Pulse Pressure", 
                step=0.1, 
                key="Pulse_Pressure_presence",
                help="Perbedaan antara tekanan sistolik dan diastolik, yang dapat menunjukkan kesehatan jantung. Nilai normal berkisar antara 30 hingga 50 mmHg."
            ),
        "Diastolic_BP":
            st.number_input(
                "Diastolic BP", 
                step=0.1, 
                key="Diastolic_BP_presence",
                help="Tekanan darah diastolik, yaitu tekanan pada arteri ketika jantung berada dalam kondisi istirahat di antara dua detak, ketika jantung mengisi kembali dengan darah. Nilai normal berkisar antara 60 hingga 80 mmHg."
            ),
        "Systolic_BP":
            st.number_input(
                "Systolic BP",  
                step=0.1, 
                key="Systolic_BP_presence",
                help="Tekanan darah sistolik, yaitu tekanan pada arteri ketika jantung berkontraksi dan memompa darah keluar ke seluruh tubuh. Nilai normal berkisar antara 90 hingga 120 mmHg."
            ),
        "Resting_HR":
            st.number_input(
                "Resting HR", 
                step=0.1, 
                key="Resting_HR_presence",
                help="Jumlah denyut jantung per menit ketika seseorang dalam kondisi istirahat penuh, yang dapat menunjukkan kesehatan jantung. Nilai normal berkisar antara 60 hingga 100 detak per menit."
            )
    }
        
        # Button to make prediction
        if st.button("Prediksi", key="predict_presence"):
            try:
                ordered_data_presence = {key: data_presence[key] for key in features_order_presence}
                
                if any(val == "" for val in ordered_data_presence.values()):
                    st.warning("Mohon lengkapi semua kolom sebelum melakukan prediksi.")
                    st.stop()
   
                response = requests.post(
                    "https://fastapicvd-production.up.railway.app/predict-presence",
                    json=ordered_data_presence
                )
                if response.status_code == 200:
                    result_presence = response.json()["prediction_presence"]
                    st.session_state.hasil_prediksi_presence = result_presence
                    st.success(f"Prediksi berhasil! Hasil: {result_presence}")

                    # Kirim hasil ke Supabase
                    try:
                        requests.post("https://fastapicvd-production.up.railway.app/save-prediction",
                         json={
                             "nama": st.session_state.get("nama"),
                             "email": st.session_state.get("email"),
                             "no_tlp": st.session_state.get("no_tlp"),
                             "target": "presence",
                             "hasil_prediksi": result_presence
                         })
                    except Exception as err:
                        st.warning(f"Gagal menyimpan hasil ke database: {err}")   
                else:
                    st.error("Gagal memproses prediksi: " + response.text)
            
            except Exception as e:
                st.error(f"Terjadi error saat koneksi ke API: {e}")

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
                del st.session_state["hasil_prediksi_presence"]
                st.rerun()