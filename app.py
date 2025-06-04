import os
import pandas as pd
import numpy as np
import streamlit as st
import requests


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
                    st.experimental_rerun() # Reload app untuk menampilkan tab prediksi
                else:
                    st.error("Reistrasi gagal: "+ response.text)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghubungi server: {e}")
                st.stop()

# Jika sudah terdaftar, tampilkan dua tab prediksi
if st.session_state['registered']:
    st.title("Menu Prediksi CVD")
    st.write("This is a web application to predict CVD using Machine Learning.")
    st.write("Please fill in the following form to make a prediction.")
    
    tab_risk, tab_presence = st.tabs(["Prediksi CVD Risk Score", "Prediksi CVD Presence"])
    
    with tab_risk:
        st.subheader("Prediksi Risiko CVD")
        st.write("Silakan isi data berikut untuk memprediksi risiko CVD.")
        
        # Input Form
        data_risk = pd.DataFrame({
        "Hypertension" : [
            st.selectbox(
                "Hypertension", 
                options=encoder_Hypertension_risk.classes_, 
                index=1, 
                key="Hypertension_risk",
                help="Tekanan darah tinggi, kondisi di mana tekanan darah dalam arteri meningkat secara kronis. Pilih 'Yes' jika ada riwayat hipertensi, 'No' jika tidak."
            )
        ],
        "ECG_Abnormality" : [
            st.selectbox(
                "ECG Abnormality",
                options=encoder_ECG_Abnormality_risk.classes_,
                index=1, 
                key="ECG_Abnormality_risk",
                help="Kelainan pada hasil elektrokardiogram (ECG), yang dapat menunjukkan masalah jantung. Pilih 'Normal' jika tidak ada kelainan; 'Arrhythmia' jika ketidaknormalan dalam irama jantung, berupa detak jantung terlalu cepat, terlalu lambat, atau tidak teratur; 'Ischemia' jika kondisi otot jantung tidak mendapatkan cukup oksigen karena aliran darah berkurang."
            )
        ],
        "Diabetes" : [
            st.selectbox(
                "Diabetes",
                options=encoder_Diabetes_risk.classes_, 
                index=0, 
                key="Diabetes_risk",
                help="Kondisi di mana tubuh tidak dapat mengatur kadar gula darah dengan baik. Pilih 'Yes' jika ada riwayat diabetes, 'No' jika tidak."
            )
        ],
        "Alcohol" : [
            st.selectbox(
                "Alcohol", 
                options=encoder_Alcohol_risk.classes_, 
                index=1, 
                key="Alcohol_risk",
                help="Konsumsi alkohol, yang dapat mempengaruhi kesehatan jantung. Pilih 'Yes' jika mengonsumsi alkohol, 'No' jika tidak."
            )
        ],
        "Previous_Stroke" : [
            st.selectbox(
                "Previous Stroke", 
                options=encoder_Previous_Stroke_risk.classes_, 
                index=0, 
                key="Previous_Stroke_risk",
                help="Riwayat stroke sebelumnya, yang dapat meningkatkan risiko CVD. Pilih 'Yes' jika ada riwayat stroke, 'No' jika tidak."
            )
        ],
        "Family_History" : [
            st.selectbox(
                "Family History", 
                options=encoder_Family_History_risk.classes_, 
                index=1, 
                key="Family_History_risk",
                help="Riwayat penyakit jantung dalam keluarga, yang dapat meningkatkan risiko CVD. Pilih 'Yes' jika ada riwayat keluarga, 'No' jika tidak."
            )
        ],
        "Insulin_Resistance" : [
            st.number_input(
                "Insulin Resistance", 
                value=4.732879, 
                step=0.1, 
                key="Insulin_Resistance_risk",
                help="Kondisi dimana sel-sel dalam tubuh tidak merespon dengan baik terhadap insulin. Tingkat resistensi insulin terjadi saat glukosa tidak bisa masuk ke dalam sel dengan efisien yang mengakibatkan peningkatan kadar glukosa dalam darah sehingga dapat mempengaruhi risiko CVD. Nilai normal berkisar antara 0 hingga 10, dengan nilai lebih tinggi menunjukkan resistensi yang lebih besar."
            )
        ],
        "Pulse_Pressure" : [
            st.number_input(
                "Pulse Pressure", 
                value=42.972956, 
                step=0.1, 
                key="Pulse_Pressure_risk",
                help="Perbedaan antara tekanan sistolik dan diastolik, yang dapat menunjukkan kesehatan jantung. Nilai normal berkisar antara 30 hingga 50 mmHg."
            )
        ],
        "Diastolic_BP" : [
            st.number_input(
                "Diastolic BP", 
                value=86.808942, 
                step=0.1, 
                key="Diastolic_BP_risk",
                help="Tekanan darah diastolik, yaitu tekanan pada arteri ketika jantung berada dalam kondisi istirahat di antara dua detak, ketika jantung mengisi kembali dengan darah. Nilai normal berkisar antara 60 hingga 80 mmHg."
            )
        ],
        "Systolic_BP" : [
            st.number_input(
                "Systolic BP", 
                value=111.648090, 
                step=0.1, 
                key="Systolic_BP_risk",
                help="Tekanan darah sistolik, yaitu tekanan pada arteri ketika jantung berkontraksi dan memompa darah keluar ke seluruh tubuh. Nilai normal berkisar antara 90 hingga 120 mmHg."
            )
        ],
        "Resting_HR" : [
            st.number_input(
                "Resting HR", 
                value=72.329284, 
                step=0.1, 
                key="Resting_HR_risk",
                help="Jumlah denyut jantung per menit ketika seseorang dalam kondisi istirahat penuh, yang dapat menunjukkan kesehatan jantung. Nilai normal berkisar antara 60 hingga 100 detak per menit."
            )
        ]
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
            
            try:
                response = requests.post(
                    "https://fastapicvd-production.up.railway.app/predict-risk",
                    json=new_data_risk.iloc[0].to_dict()
                )
                if response.status_code == 200:
                    result_risk = response.json()["prediction"]
                    st.session_state.hasil_prediksi_risk = result_risk  # Save result in session state
                else:
                    st.error("Prediksi gagal: " + response.text)
            except Exception as e:
                st.error(f"Terjadi error saat koneksi ke API: {e}")
            
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
        "Hypertension" : [
            st.selectbox(
                "Hypertension", 
                options=encoder_Hypertension_presence.classes_, 
                index=1, 
                key="Hypertension_presence",
                help="Tekanan darah tinggi, kondisi di mana tekanan darah dalam arteri meningkat secara kronis. Pilih 'Yes' jika ada riwayat hipertensi, 'No' jika tidak."
            )
        ],
        "ECG_Abnormality" : [
            st.selectbox(
                "ECG Abnormality",
                options=encoder_ECG_Abnormality_presence.classes_,
                index=0, 
                key="ECG_Abnormality_presence",
                help="Kelainan pada hasil elektrokardiogram (ECG), yang dapat menunjukkan masalah jantung. Pilih 'Normal' jika tidak ada kelainan; 'Arrhythmia' jika ketidaknormalan dalam irama jantung, berupa detak jantung terlalu cepat, terlalu lambat, atau tidak teratur; 'Ischemia' jika kondisi otot jantung tidak mendapatkan cukup oksigen karena aliran darah berkurang."
            )
        ],
        "Diabetes" : [
            st.selectbox(
                "Diabetes",
                options=encoder_Diabetes_presence.classes_, 
                index=1, 
                key="Diabetes_presence",
                help="Kondisi di mana tubuh tidak dapat mengatur kadar gula darah dengan baik. Pilih 'Yes' jika ada riwayat diabetes, 'No' jika tidak."
            )
        ],
        "Alcohol" : [
            st.selectbox(
                "Alcohol", 
                options=encoder_Alcohol_presence.classes_, 
                index=0, 
                key="Alcohol_presence",
                help="Konsumsi alkohol, yang dapat mempengaruhi kesehatan jantung. Pilih 'Yes' jika mengonsumsi alkohol, 'No' jika tidak."
            )
        ],
        "Previous_Stroke" : [
            st.selectbox(
                "Previous Stroke", 
                options=encoder_Previous_Stroke_presence.classes_, 
                index=1, 
                key="Previous_Stroke_presence",
                help="Riwayat stroke sebelumnya, yang dapat meningkatkan risiko CVD. Pilih 'Yes' jika ada riwayat stroke, 'No' jika tidak."
            )
        ],
        "Family_History" : [
            st.selectbox(
                "Family History", 
                options=encoder_Family_History_presence.classes_, 
                index=1, 
                key="Family_History_presence",
                help="Riwayat penyakit jantung dalam keluarga, yang dapat meningkatkan risiko CVD. Pilih 'Yes' jika ada riwayat keluarga, 'No' jika tidak."
            )
        ],
        "CVD_Risk_Score" : [
            st.selectbox(
                "CVD Risk Score", 
                options=encoder_CVD_Risk_Score_presence.classes_, 
                index=1, 
                key="CVD_Risk_Score_presence",
                help="Skor tingkat risiko seseorang untuk mengembangkan penyakit kardiovaskular di masa depan. Pilih 'Low' jika skor risiko rendah, 'Moderate' jika sedang, dan 'High' jika tinggi."
            )
        ],
        "Insulin_Resistance" : [
            st.number_input(
                "Insulin Resistance", 
                value=6.441541, 
                step=0.1, 
                key="Insulin_Resistance_presence",
                help="Kondisi dimana sel-sel dalam tubuh tidak merespon dengan baik terhadap insulin. Tingkat resistensi insulin terjadi saat glukosa tidak bisa masuk ke dalam sel dengan efisien yang mengakibatkan peningkatan kadar glukosa dalam darah sehingga dapat mempengaruhi risiko CVD. Nilai normal berkisar antara 0 hingga 10, dengan nilai lebih tinggi menunjukkan resistensi yang lebih besar."
            )
        ],
        "Pulse_Pressure" : [
            st.number_input(
                "Pulse Pressure", 
                value=33.438115, 
                step=0.1, 
                key="Pulse_Pressure_presence",
                help="Perbedaan antara tekanan sistolik dan diastolik, yang dapat menunjukkan kesehatan jantung. Nilai normal berkisar antara 30 hingga 50 mmHg."
            )
        ],
        "Diastolic_BP" : [
            st.number_input(
                "Diastolic BP", 
                value=73.272788, 
                step=0.1, 
                key="Diastolic_BP_presence",
                help="Tekanan darah diastolik, yaitu tekanan pada arteri ketika jantung berada dalam kondisi istirahat di antara dua detak, ketika jantung mengisi kembali dengan darah. Nilai normal berkisar antara 60 hingga 80 mmHg."
            )
        ],
        "Systolic_BP" : [
            st.number_input(
                "Systolic BP", 
                value=116.245744, 
                step=0.1, 
                key="Systolic_BP_presence",
                help="Tekanan darah sistolik, yaitu tekanan pada arteri ketika jantung berkontraksi dan memompa darah keluar ke seluruh tubuh. Nilai normal berkisar antara 90 hingga 120 mmHg."
            )
        ],
        "Resting_HR" : [
            st.number_input(
                "Resting HR", 
                value=60.970855, 
                step=0.1, 
                key="Resting_HR_presence",
                help="Jumlah denyut jantung per menit ketika seseorang dalam kondisi istirahat penuh, yang dapat menunjukkan kesehatan jantung. Nilai normal berkisar antara 60 hingga 100 detak per menit."
            )
        ]
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

            try:
                response = requests.post(
                    "https://fastapicvd-production.up.railway.app/predict-presence",
                    json=new_data_presence.iloc[0].to_dict()
                )
                if response.status_code == 200:
                    result_presence = response.json()["prediction"]
                    st.session_state.hasil_prediksi_presence = result_presence  # Save result in session state
                else:
                    st.error("Prediksi gagal: " + response.text)
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
                st.session_state.hasil_prediksi_presence = None
                st.experimental_rerun()